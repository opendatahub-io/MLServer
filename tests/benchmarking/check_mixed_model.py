import asyncio
import hashlib
import json
import logging
import math
import socket
import sqlite3
import struct
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, closing
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import aiohttp
import pytest
import uvicorn
from aiohttp import web
from aiohttp.test_utils import TestServer
from grpc import StatusCode, aio

from benchmarking.mixed_model.report import summary
from benchmarking.mixed_model.run import Scenario, check_output, database, main, run
from benchmarking.mixed_model.transport import GrpcCalls, failure
from mlserver import Settings
from mlserver.grpc import GRPCServer
from mlserver.grpc import dataplane_pb2 as pb
from mlserver.grpc.converters import ModelInferRequestConverter
from mlserver.grpc.dataplane_pb2_grpc import GRPCInferenceServiceStub
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.logging import ModelLoggerFormatter
from mlserver.parallel import InferencePoolRegistry
from mlserver.registry import MultiModelRegistry
from mlserver.rest.app import create_app
from mlserver.settings import ModelParameters, ModelSettings
from mlserver.types import ResponseOutput


def scenario(name="A", version="1", **kwargs):
    tensor = dict(name="total", shape=[1, 1], datatype="FP32", data=[3])
    config = dict(
        rate=4,
        duration_s=1,
        warmup_s=0,
        deadline_s=1,
        concurrency=1,
        seed=1,
        deployment={"workers": 1},
        initial_load=[],
        actions=[],
        models=[
            dict(
                name=name,
                version=version,
                artifact_sha256="0" * 64,
                request={
                    "inputs": [
                        dict(name="input-0", shape=[1, 2], datatype="FP32", data=[1, 2])
                    ]
                },
                expected=[tensor],
            )
        ],
    )
    return Scenario.model_validate(dict(config, **kwargs))


def reply(model, version, request_id, value):
    return pb.ModelInferResponse(
        id=request_id,
        model_name=model,
        model_version=version,
        outputs=[
            pb.ModelInferResponse.InferOutputTensor(
                name="total",
                shape=[1, 1],
                datatype="FP32",
                contents=pb.InferTensorContents(fp32_contents=[value]),
            )
        ],
    )


@pytest.mark.parametrize(
    "model,version,value", [("B", "1", 3), ("A", "2", 3), ("A", "1", 4)]
)
def test_distinguishes_model_version_and_result(model, version, value):
    target = scenario().models[0]
    check_output(reply("A", "1", "id", 3), target, "id")
    with pytest.raises(ValueError):
        check_output(reply(model, version, "id", value), target, "id")


@pytest.mark.parametrize("protocol", ["rest", "grpc"])
@pytest.mark.parametrize(
    "datatype,expected,wrong,field",
    [
        ("UINT64", [0, 2**64 - 1], [0, 2**64 - 2], "uint64_contents"),
        ("BYTES", ["héllo", "a\0"], ["héllo", "a"], "bytes_contents"),
    ],
)
def test_exact_tensor_values(protocol, datatype, expected, wrong, field):
    target = scenario().models[0]
    output = target.expected[0]
    output.datatype = datatype
    output.shape = [2]
    output.data.root = expected

    def response(values):
        tensor = dict(name=output.name, shape=[2], datatype=datatype)
        identity = dict(id="id", model_name="A", model_version="1")
        if protocol == "rest":
            return json.dumps(
                dict(**identity, outputs=[dict(**tensor, data=values)])
            ).encode()
        if datatype == "BYTES":
            values = [s.encode("utf-8") for s in values]
        return pb.ModelInferResponse(
            **identity, outputs=[dict(**tensor, contents={field: values})]
        )

    check_output(response(expected), target, "id")
    with pytest.raises(ValueError, match="incorrect values"):
        check_output(response(wrong), target, "id")


@pytest.mark.parametrize(
    "datatype,expected,invalid", [("FP32", 0.0, False), ("BOOL", True, 1)]
)
def test_output_contents_must_match_declared_datatype(datatype, expected, invalid):
    target = scenario().models[0]
    output = target.expected[0]
    output.datatype = datatype
    output.data.root = [expected]
    tensor = dict(name=output.name, datatype=datatype, shape=output.shape)
    identity = dict(id="id", model_name="A", model_version="1")
    valid = dict(**identity, outputs=[dict(**tensor, data=[expected])])
    check_output(json.dumps(valid).encode(), target, "id")
    invalid_rest = dict(**identity, outputs=[dict(**tensor, data=[invalid])])
    with pytest.raises(ValueError, match="contents disagree with datatype"):
        check_output(json.dumps(invalid_rest).encode(), target, "id")
    invalid_grpc = pb.ModelInferResponse(
        **identity, outputs=[dict(**tensor, contents={"int_contents": [int(expected)]})]
    )
    with pytest.raises(ValueError, match="contents disagree with datatype"):
        check_output(invalid_grpc, target, "id")


def test_grpc_template_encodes_scenario_text():
    target = scenario().models[0]
    tensor = target.request.inputs[0]
    tensor.datatype = "BYTES"
    tensor.data.root = ["héllo", "a\0"]
    template = GrpcCalls(Mock(spec=GRPCInferenceServiceStub), [target]).templates[0]
    decoded = ModelInferRequestConverter.to_types(template)
    assert decoded.inputs[0].data.root == ["héllo".encode(), b"a\0"]
    assert tensor.data.root == ["héllo", "a\0"]


def test_grpc_raw_reply_requires_a_buffer_even_for_an_empty_output():
    target = scenario().models[0]
    target.expected.append(
        ResponseOutput(name="empty", shape=[0], datatype="FP32", data=[])
    )
    response = reply("A", "1", "id", 3)
    response.outputs[0].ClearField("contents")
    response.outputs.add(name="empty", shape=[0], datatype="FP32")
    response.raw_output_contents.extend([struct.pack("<f", 3), b""])
    check_output(response, target, "id")

    del response.raw_output_contents[-1]
    with pytest.raises(ValueError, match="one buffer per output"):
        check_output(response, target, "id")


@pytest.mark.parametrize(
    "include_typed,raw_buffers",
    [
        pytest.param(False, [b""], id="truncated"),
        pytest.param(True, [struct.pack("<f", 4)], id="typed-and-raw"),
        pytest.param(False, [struct.pack("<f", 3), struct.pack("<f", 4)], id="surplus"),
    ],
)
async def test_malformed_grpc_reply_is_recorded_and_traffic_continues(
    tmp_path, include_typed, raw_buffers
):
    class Client:
        async def ModelInfer(self, request, timeout):
            response = reply("A", "1", request.id, 3)
            if request.id.endswith(".1"):
                if not include_typed:
                    response.outputs[0].ClearField("contents")
                response.raw_output_contents.extend(raw_buffers)
            return response

    with closing(database(tmp_path / "run.db")) as db:
        await run(scenario(), Client(), db, "test")
        result = summary(db)
        assert result["status"] == "complete"
        window = result["models"][0]["windows"]["all"]
        assert window["outcomes"] == {"correct": 3, "wrong_output": 1}
        assert window["accepted_lower_bound_from_replies"] == 3


@pytest.mark.parametrize("delay", [0, 0.15])
async def test_measurement_waits_for_warmup_interval_and_replies(tmp_path, delay):
    clock = asyncio.get_running_loop().time
    events = {}

    class Client:
        async def ModelInfer(self, request, timeout):
            if ".warmup." in request.id:
                await asyncio.sleep(delay)
                events["warmup_return"] = clock()
            else:
                events["measurement_start"] = clock()
            return reply("A", "1", request.id, 3)

    config = scenario(rate=1, warmup_s=0.1, duration_s=0.05)
    with closing(database(tmp_path / "run.db")) as db:
        start = clock()
        await run(config, Client(), db, "test")
    assert events["measurement_start"] >= start + config.warmup_s
    assert events["measurement_start"] >= events["warmup_return"]


@pytest.mark.parametrize(
    "duration,expected", [(0.14, 14), (math.nextafter(0.14, 1), 15)]
)
async def test_arrivals_match_the_report_window(tmp_path, duration, expected):
    class Client:
        async def ModelInfer(self, request, timeout):
            return reply("A", "1", request.id, 3)

    with closing(database(tmp_path / "run.db")) as db:
        await run(scenario(rate=100, duration_s=duration), Client(), db, "test")
        count, last = db.execute("SELECT COUNT(*), MAX(due) FROM attempt").fetchone()
        assert (
            count == summary(db)["models"][0]["windows"]["all"]["offered"] == expected
        )
        assert last < duration
        db.execute("DROP TABLE completed_run")  # Original database schema.
        legacy = summary(db)
        assert legacy["status"] == "unrecorded"
        assert (
            legacy["models"][0]["windows"]["all"]["correct_responses_per_s"] is not None
        )


async def test_generator_accounts_for_arrivals_while_response_is_delayed(tmp_path):
    release = asyncio.Event()

    class Client:
        async def ModelInfer(self, request, timeout):
            await release.wait()
            return reply(request.model_name, request.model_version, request.id, 3)

    db = database(tmp_path / "run.db")
    job = asyncio.create_task(run(scenario(), Client(), db, "test"))
    try:
        async with asyncio.timeout(8):
            while not db.execute(
                "SELECT count(*)=3 FROM attempt WHERE outcome LIKE 'generator_%'"
            ).fetchone()[0]:
                await asyncio.sleep(0.01)
            release.set()
            await job
        all_results = summary(db)["models"][0]["windows"]["all"]
        assert all_results["offered"] == 4
        assert all_results["outcomes"] == {"correct": 1, "generator_full": 3}
        assert (
            db.execute(
                "SELECT COUNT(*) FROM attempt WHERE submitted IS NOT NULL"
            ).fetchone()[0]
            == 1
        )
    finally:
        release.set()
        await job
        db.close()


async def test_resource_sampling_recovers_and_keeps_valid_gpus(
    tmp_path, monkeypatch, caplog
):
    smi = tmp_path / "nvidia-smi"
    smi.write_text(
        '#!/bin/sh\nif [ ! -f "$0.seen" ]; then touch "$0.seen"; exit 1; fi\n'
        "printf 'gpu-a, [N/A]\\ngpu-b, 7\\n'\n"
    )
    smi.chmod(0o755)
    monkeypatch.setattr("benchmarking.mixed_model.run.shutil.which", lambda _: str(smi))

    class Client:
        async def ModelInfer(self, request, timeout):
            return reply(request.model_name, request.model_version, request.id, 3)

    db = database(tmp_path / "run.db")
    try:
        await run(scenario(duration_s=1.25), Client(), db, "test", local_resources=True)
        assert summary(db)["models"][0]["windows"]["all"]["outcomes"] == {"correct": 5}
        assert db.execute(
            "SELECT DISTINCT subject, bytes FROM resource WHERE source='gpu_used'"
        ).fetchall() == [("gpu-b", 7 * 1024**2)]
        assert "nvidia-smi exited 1" in caplog.text
        assert "gpu-a, [N/A]" in caplog.text
    finally:
        db.close()


async def test_failed_run_preserves_recorded_attempts(tmp_path):
    class Client:
        async def ModelInfer(self, request, timeout):
            if request.id.endswith(".1"):
                raise RuntimeError("interrupted measurement")
            return reply(request.model_name, request.model_version, request.id, 3)

    path = tmp_path / "run.db"
    db = database(path)
    try:
        with pytest.raises(ExceptionGroup):
            await run(scenario(), Client(), db, "test")
    finally:
        db.close()
    with closing(sqlite3.connect(path)) as saved:
        assert saved.execute("SELECT address FROM run").fetchall() == [("test",)]
        assert saved.execute("SELECT outcome FROM attempt").fetchall() == [("correct",)]
        assert summary(saved)["status"] == "incomplete"
    report = subprocess.run(
        [sys.executable, "-m", "benchmarking.mixed_model.report", str(path)],
        capture_output=True,
        text=True,
    )
    assert report.returncode == 1
    result = json.loads(report.stdout)
    assert result["status"] == "incomplete"
    assert result["models"][0]["windows"]["all"]["correct_responses_per_s"] is None
    with closing(sqlite3.connect(path)) as saved:
        saved.execute("DROP TABLE completed_run")
        assert summary(saved)["status"] == "incomplete"


async def _gate_worker(model, directory):
    # Use the normal worker and model. Only the test controls when predict returns.
    logger = logging.getLogger("mlserver.parallel.execution")
    handler = logging.FileHandler(directory / "worker.log")
    logger.addHandler(handler)
    structured = logging.FileHandler(directory / "structured.log")
    structured.setFormatter(ModelLoggerFormatter(Settings(use_structured_logging=True)))
    logger.addHandler(structured)
    logger.setLevel(logging.DEBUG)
    predict = model.predict

    async def gated(request):
        (directory / "entered").touch()
        async with asyncio.timeout(12):
            while not (directory / "release").exists():
                await asyncio.sleep(0.01)
        return await predict(request)

    setattr(model, "predict", gated)
    return model


@pytest.fixture
async def model_registry(
    settings, sum_model_settings, tmp_path, prometheus_registry
) -> AsyncIterator[MultiModelRegistry]:
    settings.parallel_workers = 1
    sum_model_settings.max_batch_size = 0
    pool = InferencePoolRegistry(
        settings, on_worker_load=[partial(_gate_worker, directory=tmp_path)]
    )
    registry = MultiModelRegistry(
        model_initialiser=pool.model_initialiser,
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
    )
    async with AsyncExitStack() as cleanup:
        cleanup.push_async_callback(pool.close)
        await pool.start()
        await registry.load(sum_model_settings)
        cleanup.push_async_callback(registry.unload, sum_model_settings.name)
        cleanup.callback((tmp_path / "release").touch)
        yield registry


@pytest.fixture(params=["grpc", "rest"])
async def benchmark_client(
    request, settings, data_plane, model_repository_handlers, unused_tcp_port
) -> AsyncIterator[GRPCInferenceServiceStub | aiohttp.ClientSession]:
    if request.param == "grpc":
        settings.grpc_port = unused_tcp_port
        grpc_server = GRPCServer(
            settings, data_plane, model_repository_handlers
        )._create_server()
        await grpc_server.start()
        try:
            async with aio.insecure_channel(f"127.0.0.1:{unused_tcp_port}") as channel:
                yield GRPCInferenceServiceStub(channel)
        finally:
            await grpc_server.stop(grace=None)
        return

    app = create_app(settings, data_plane, model_repository_handlers)
    server = uvicorn.Server(
        uvicorn.Config(app, log_config=None, access_log=False, lifespan="off")
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{sock.getsockname()[1]}"
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(8):
                while not server.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(0.01)
            async with aiohttp.ClientSession(address) as client:
                yield client
        finally:
            server.should_exit = True
            await serving


async def test_client_deadline_precedes_real_worker_completion(
    tmp_path: Path, benchmark_client, sum_model_settings
):
    config = scenario(
        sum_model_settings.name,
        sum_model_settings.version,
        rate=1,
        duration_s=0.1,
        deadline_s=1,
    )
    db = database(tmp_path / "run.db")
    task = asyncio.create_task(run(config, benchmark_client, db, "loopback"))
    log = tmp_path / "worker.log"
    try:
        async with asyncio.timeout(8):
            while not (tmp_path / "entered").exists():
                await asyncio.sleep(0.01)
            await task
            events = summary(db, log)["observed_worker_events"]
            assert [(e["client_outcome"], e["worker_event"]) for e in events] == [
                ("deadline", "started")
            ]
            (tmp_path / "release").touch()
            while '"event": "returned"' not in log.read_text():
                await asyncio.sleep(0.01)
            assert {
                e["worker_event"] for e in summary(db, log)["observed_worker_events"]
            } == {"started", "returned"}
            assert summary(db, tmp_path / "structured.log") == summary(db, log)
    finally:
        (tmp_path / "release").touch()
        await task
        db.close()


@pytest.mark.parametrize(
    "protocol,limit",
    [("grpc", "0"), ("grpc", "-1"), ("grpc", str(2**31)), ("rest", "1024")],
)
async def test_invalid_grpc_receive_limit(protocol, limit, tmp_path, capsys):
    db_path = tmp_path / "run.sqlite"
    args = ["scenario.json", str(db_path), "--protocol", protocol]
    with pytest.raises(SystemExit) as error:
        await main([*args, "--grpc-max-receive-message-bytes", limit])
    assert error.value.code == 2
    assert "--grpc-max-receive-message-bytes" in capsys.readouterr().err
    assert not db_path.exists()


def test_grpc_server_rejection_keeps_its_diagnostic():
    error = aio.AioRpcError(
        StatusCode.RESOURCE_EXHAUSTED,
        aio.Metadata(),
        aio.Metadata(),
        "server queue full",
    )
    assert failure(error) == ("rejected", "RESOURCE_EXHAUSTED: server queue full")


async def test_grpc_large_onnx_response(
    tmp_path, settings, model_repository, prometheus_registry, unused_tcp_port
):
    onnx = pytest.importorskip("onnx")
    runtime = pytest.importorskip("mlserver_onnx")
    # Expand a 16-byte input into more than 4 MiB of FP64 output.
    rows = 2**18 + 1
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("Expand", ["input-0", "shape"], ["y"])],
        "large-reply",
        [helper.make_tensor_value_info("input-0", onnx.TensorProto.DOUBLE, [1, 2])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.DOUBLE, [rows, 2])],
        [helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [rows, 2])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
        ),
        path,
    )
    registry = MultiModelRegistry()
    model = await registry.load(
        ModelSettings(
            name="large-reply",
            implementation=runtime.OnnxModel,
            parameters=ModelParameters(
                uri=str(path),
                version="1",
                extra={"providers": ["CPUExecutionProvider"]},
            ),
        )
    )
    settings.grpc_port = unused_tcp_port
    settings.grpc_max_message_length = 8 * 1024**2
    server = GRPCServer(
        settings,
        DataPlane(settings, registry),
        ModelRepositoryHandlers(model_repository, registry),
    )._create_server()
    await server.start()
    try:
        config = scenario(model.name, rate=1, duration_s=0.1, deadline_s=30)
        target = config.models[0]
        target.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        target.request.inputs[0].datatype = "FP64"
        target.expected = [
            ResponseOutput(
                name="predict", shape=[rows, 2], datatype="FP64", data=[1, 2] * rows
            )
        ]
        scenario_path = tmp_path / "scenario.json"
        scenario_path.write_text(config.model_dump_json())
        args = ["--address", f"127.0.0.1:{unused_tcp_port}", str(scenario_path)]
        attempt = "SELECT outcome, detail, verified_reply FROM attempt"
        default_db = tmp_path / "default.sqlite"
        with pytest.raises(SystemExit) as error:
            await main([*args, str(default_db)])
        assert error.value.code == 1
        with closing(sqlite3.connect(default_db)) as db:
            ((outcome, detail, verified),) = db.execute(attempt)
            assert (outcome, verified) == ("client_receive_limit", 0)
            assert (
                "RESOURCE_EXHAUSTED: CLIENT: Received message larger than max" in detail
            )
            assert "4194304" in detail

        configured_db = tmp_path / "configured.sqlite"
        await main(
            [
                *args,
                str(configured_db),
                "--grpc-max-receive-message-bytes",
                str(8 * 1024**2),
            ]
        )
        with closing(sqlite3.connect(configured_db)) as db:
            assert db.execute(attempt).fetchall() == [("correct", "", 1)]
            assert (
                summary(db)["deployment"]["grpc_max_receive_message_bytes"]
                == 8 * 1024**2
            )
    finally:
        await server.stop(grace=None)
        await registry.unload(model.name)


async def test_rest_records_management_status_and_inference_errors(tmp_path):
    async def manage(request):
        return web.Response(status=404)

    async def infer(request):
        body = await request.json()
        assert body["inputs"][0]["data"] == [1, 2]
        seq = int(body["id"].rsplit(".", 1)[-1])
        if seq == 0:
            return web.Response(status=429)
        if seq == 1:
            return web.Response(status=503)
        if seq == 2:
            return web.Response(text="maintenance")
        return web.json_response(
            {
                "id": "unrelated" if seq == 4 else body["id"],
                "model_name": "B" if seq == 5 else "A",
                "model_version": "1",
                "outputs": [
                    {
                        "name": "total",
                        "datatype": "FP32",
                        "shape": [1, 1],
                        "data": [{3: "invalid-float", 6: 4}.get(seq, 3)],
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_post("/v2/repository/models/missing/unload", manage)
    app.router.add_post("/v2/models/A/versions/1/infer", infer)
    config = scenario(
        duration_s=2,
        actions=[
            dict(
                at_s=0, operation="unload", model="missing", expected_status="HTTP_404"
            )
        ],
    )
    db = database(tmp_path / "rest.sqlite")
    try:
        async with TestServer(app) as server:
            address = str(server.make_url("/"))
            async with aiohttp.ClientSession(address) as client:
                await run(config, client, db, address)
        result = summary(db)
        assert result["deployment"]["protocol"] == "rest"
        assert result["actions"][0]["status"] == result["actions"][0]["expected"]
        window = result["models"][0]["windows"]["all"]
        assert window["outcomes"] == {
            "rejected": 1,
            "http_error": 1,
            "wrong_output": 5,
            "correct": 1,
        }
        assert window["accepted_lower_bound_from_replies"] == 2
        # An archived run cannot distinguish the five wrong_output cases.
        db.execute(
            "CREATE TEMP VIEW attempt AS SELECT id,phase,model,version,due,"
            "submitted,finished,outcome,detail FROM main.attempt"
        )
        legacy = summary(db)["models"][0]["windows"]["all"]
        assert legacy["outcomes"] == window["outcomes"]
        assert legacy["accepted_lower_bound_from_replies"] == 1
    finally:
        db.close()
