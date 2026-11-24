"""Qualify the session and operator placement inside each MLServer worker."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest

from mlserver import MLModel, ModelSettings, Settings
from mlserver.errors import ModelLoadError
from mlserver.parallel.pool import InferencePool
from mlserver.registry import MultiModelRegistry
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver_onnx import OnnxModel


async def _record_first_prediction(model: MLModel) -> MLModel:
    assert isinstance(model, OnnxModel) and model._onnx_settings is not None
    options = model._onnx_settings.session_options
    assert options is not None
    pid = os.getpid()
    prefix = f"{options['profile_file_prefix']}-{pid}"
    options["profile_file_prefix"] = prefix
    predict = model.predict

    async def record(request: InferenceRequest) -> InferenceResponse:
        response = await predict(request)
        session = model._model
        settings = session.get_session_options()
        receipt = {
            "pid": pid,
            "providers": session.get_providers(),
            "options": session.get_provider_options(),
            "threads": [settings.intra_op_num_threads, settings.inter_op_num_threads],
            "trace": session.end_profiling(),
        }
        Path(prefix + ".json").write_text(json.dumps(receipt))
        setattr(model, "predict", predict)
        return response

    setattr(model, "predict", record)
    return model


@pytest.mark.parametrize(
    "provider",
    [
        "CPUExecutionProvider",
        pytest.param("CUDAExecutionProvider", marks=pytest.mark.cuda),
    ],
)
async def test_worker_profile_after_restart(
    model_settings: ModelSettings,
    inference_request: InferenceRequest,
    tmp_path: Path,
    provider: str,
    cuda_device_id: int,
) -> None:
    extra: dict[str, Any] = {
        "providers": [provider],
        "session_options": {
            "enable_profiling": True,
            "profile_file_prefix": str(tmp_path / "worker"),
            "intra_op_num_threads": 2,
            "inter_op_num_threads": 1,
        },
    }
    if provider == "CUDAExecutionProvider":
        from .test_cuda import HAS_CUDA

        if not HAS_CUDA:
            pytest.skip("Requires a working CUDA provider")
        extra["provider_options"] = {"device_id": cuda_device_id}
        extra["session_config_entries"] = {"session.disable_cpu_ep_fallback": "1"}
    assert model_settings.parameters is not None
    model_settings.parameters.extra = extra
    pool = InferencePool(
        Settings(parallel_workers=2), on_worker_load=[_record_first_prediction]
    )
    expected = np.asarray(inference_request.inputs[0].data.root) + 1
    try:
        await pool.start()
        model = await pool.load_model(OnnxModel(model_settings))

        async def check_workers() -> set[int]:
            pids = set(pool._workers)
            assert len(pids) == 2
            for _ in pids:
                result = await model.predict(inference_request)
                np.testing.assert_allclose(result.outputs[0].data.root, expected)
            for pid in pids:
                receipt = json.loads((tmp_path / f"worker-{pid}.json").read_text())
                assert receipt["pid"] == pid and receipt["threads"] == [2, 1]
                assert receipt["providers"][0] == provider
                if provider == "CUDAExecutionProvider":
                    assert (
                        int(receipt["options"][provider]["device_id"]) == cuda_device_id
                    )
                trace = json.loads(Path(receipt["trace"]).read_text())
                assert {
                    e["args"]["provider"]
                    for e in trace
                    if e.get("pid") == pid
                    and e.get("args", {}).get("op_name") == "Add"
                    and "provider" in e["args"]
                } == {provider}
            return pids

        before = await asyncio.wait_for(check_workers(), 60)
        worker = next(iter(pool._workers.values()))
        worker.kill()
        await asyncio.to_thread(worker.join, 10)
        assert worker.pid is not None and worker.exitcode is not None
        await pool.on_worker_stop(worker.pid, worker.exitcode)
        await asyncio.wait_for(asyncio.gather(*pool._restart_tasks), 60)
        after = await asyncio.wait_for(check_workers(), 60)
        assert before & after == before - {worker.pid}
    finally:
        await pool.close()


@pytest.mark.parametrize(
    "cuda_options,error",
    [
        ([{"device_id": -1}], "device_id"),
        ([{"device_id": " 0 "}], "device_id"),
        ([{}, {"device_id": "cuda:0"}], "device_id"),
        ([{}, {"device_id": 1}], "device_id"),
        ([{}, {"enable_cuda_graph": "1"}], "enable_cuda_graph"),
        ([{"enable_cuda_graph": "0"}, {"enable_cuda_graph": "1"}], "enable_cuda_graph"),
        ([{"enable_cuda_graph": None}], "enable_cuda_graph"),
        ([{"enable_cuda_graph": "None"}], "enable_cuda_graph"),
        (
            [{"enable_cuda_graph": None}, {"enable_cuda_graph": "1"}],
            "enable_cuda_graph",
        ),
    ],
)
async def test_invalid_selection_is_not_published(
    model_settings: ModelSettings, cuda_options: list[dict[str, Any]], error: str
) -> None:
    assert model_settings.parameters is not None
    model_settings.parameters.extra = {
        "providers": ["CUDAExecutionProvider"] * len(cuda_options)
        + ["CPUExecutionProvider"],
        "provider_options": [*cuda_options, {}],
    }
    registry = MultiModelRegistry()
    with pytest.raises(ModelLoadError, match=error):
        await registry.load(model_settings)
    assert await registry.get_models() == []


@pytest.mark.parametrize(
    "provider",
    [
        "CPUExecutionProvider",
        pytest.param("CUDAExecutionProvider", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "cuda_options",
    [
        [{"enable_cuda_graph": None}, {"enable_cuda_graph": "1"}],
        [{"enable_cuda_graph": "None"}],
    ],
    ids=["repeated-null", "standalone-invalid-string"],
)
async def test_invalid_graph_replacement_preserves_original(
    model_settings: ModelSettings,
    inference_request: InferenceRequest,
    provider: str,
    cuda_device_id: int,
    cuda_options: list[dict[str, Any]],
) -> None:
    if provider == "CUDAExecutionProvider":
        from .test_cuda import HAS_CUDA

        if not HAS_CUDA:
            pytest.skip("Requires a working CUDA provider")

    assert model_settings.parameters is not None
    original_settings = model_settings.model_copy(deep=True)
    assert original_settings.parameters is not None
    original_settings.parameters.extra = {"providers": [provider]}
    if provider == "CUDAExecutionProvider":
        original_settings.parameters.extra.update(
            {
                "provider_options": [{"device_id": cuda_device_id}],
                "session_config_entries": {"session.disable_cpu_ep_fallback": "1"},
            }
        )

    registry = MultiModelRegistry()
    original = await registry.load(original_settings)
    try:
        assert isinstance(original, OnnxModel) and original.ready
        assert original._model.get_providers()[0] == provider
        before = await original.predict(inference_request)

        replacement = model_settings.model_copy(deep=True)
        assert replacement.parameters is not None
        replacement.parameters.extra = {
            "providers": ["CUDAExecutionProvider"] * len(cuda_options)
            + ["CPUExecutionProvider"],
            "provider_options": [*cuda_options, {}],
        }
        with pytest.raises(ModelLoadError, match="enable_cuda_graph"):
            await registry.load(replacement)

        assert await registry.get_model(original.name, original.version) is original
        assert await registry.get_models() == [original]
        assert original.ready
        after = await original.predict(inference_request)
        np.testing.assert_allclose(
            after.outputs[0].data.root, before.outputs[0].data.root
        )
    finally:
        await registry.unload(original.name)


@pytest.mark.parametrize(
    "extra",
    [
        {"providers": ["CUDAExecutionProvider"]},
        {
            "providers": ["CUDAExecutionProvider"],
            "provider_options": {"enable_cuda_graph": "1"},
            "run_options": {"log_severity_level": 3},
        },
        {
            "providers": ["CUDAExecutionProvider"] * 2,
            "provider_options": [{}, {"enable_cuda_graph": "0"}],
        },
    ],
)
async def test_cpu_host_fallback_policy(
    model_settings: ModelSettings, inference_request: InferenceRequest, extra: dict
) -> None:
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("Requires an installation without the CUDA provider")
    assert model_settings.parameters is not None
    model_settings.parameters.extra = extra
    registry = MultiModelRegistry()
    model = await registry.load(model_settings)
    try:
        assert isinstance(model, OnnxModel) and model._model.get_providers() == [
            "CPUExecutionProvider"
        ]
        response = await model.predict(inference_request)
        np.testing.assert_allclose(
            response.outputs[0].data.root,
            np.asarray(inference_request.inputs[0].data.root) + 1,
        )
    finally:
        await registry.unload(model.name)
    model_settings.parameters.extra["session_config_entries"] = {
        "session.disable_cpu_ep_fallback": "1"
    }
    with pytest.raises(ModelLoadError):
        await registry.load(model_settings)
    assert await registry.get_models() == []
