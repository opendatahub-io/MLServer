"""Run an open-loop mixed-model scenario over V2 gRPC or REST."""

import argparse
import asyncio
import json
import logging
import math
import random
import shutil
import sqlite3
import struct
import uuid
from collections.abc import Callable
from itertools import count
from pathlib import Path
from typing import Annotated, Literal, Self

import aiohttp
import grpc
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from mlserver import types
from mlserver.codecs.string import encode_str
from mlserver.errors import InferenceError
from mlserver.grpc import dataplane_pb2 as pb
from mlserver.grpc.converters import _FIELDS as _GRPC_CONTENT_FIELDS
from mlserver.grpc.converters import ModelInferResponseConverter
from mlserver.grpc.dataplane_pb2_grpc import GRPCInferenceServiceStub

from .report import summary
from .transport import GrpcCalls, RestCalls, failure

logger = logging.getLogger(__name__)
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Target(Config):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    weight: PositiveInt = 1
    start_s: Nonnegative = 0
    request: types.InferenceRequest
    expected: list[types.ResponseOutput] = Field(min_length=1)
    rtol: Nonnegative = 1e-5
    atol: Nonnegative = 1e-7

    @model_validator(mode="after")
    def check_expected(self) -> Self:
        if len({output.name for output in self.expected}) != len(self.expected):
            raise ValueError("expected output names must be unique")
        for output in self.expected:
            if len(output.data.root) != math.prod(output.shape):
                raise ValueError(
                    f"expected data length disagrees with {output.name} shape"
                )
        return self


class Action(Config):
    at_s: Nonnegative
    operation: Literal["load", "unload"]
    model: str = Field(min_length=1)
    expected_status: str = "OK"


class Scenario(Config):
    rate: PositiveFloat
    duration_s: PositiveFloat
    warmup_s: Nonnegative
    concurrency: PositiveInt
    deadline_s: PositiveFloat
    management_timeout_s: PositiveFloat = 60
    seed: int
    deployment: dict = Field(min_length=1)
    models: list[Target] = Field(min_length=1)
    initial_load: list[str]
    actions: list[Action]

    @model_validator(mode="after")
    def check_schedule(self) -> Self:
        if len({(m.name, m.version) for m in self.models}) != len(self.models):
            raise ValueError("model/version targets must be unique")
        if not any(m.start_s == 0 for m in self.models):
            raise ValueError("at least one model must receive traffic from time zero")
        if any(m.start_s >= self.duration_s for m in self.models):
            raise ValueError("model start must precede the measurement end")
        if any(a.at_s >= self.duration_s for a in self.actions):
            raise ValueError("action must start within the measurement window")
        return self


class _IncorrectValues(ValueError):
    """A well-formed reply matched the request, model and tensor metadata."""


def check_output(
    reply: pb.ModelInferResponse | bytes, target: Target, request_id: str
) -> None:
    if isinstance(reply, bytes):
        response = types.InferenceResponse.model_validate_json(reply)
    else:
        if reply.raw_output_contents and (
            len(reply.raw_output_contents) != len(reply.outputs)
            or any(output.HasField("contents") for output in reply.outputs)
        ):
            raise ValueError(
                "raw gRPC replies require one buffer per output and no typed contents"
            )
        for tensor in reply.outputs:
            field = _GRPC_CONTENT_FIELDS.get(types.Datatype(tensor.datatype))
            if any(desc.name != field for desc, _ in tensor.contents.ListFields()):
                raise ValueError(f"{tensor.name}: contents disagree with datatype")
        try:
            response = ModelInferResponseConverter.to_types(reply)
        except (struct.error, IndexError, OverflowError, InferenceError) as exc:
            raise ValueError(f"invalid gRPC output: {exc}") from exc
    if (response.id, response.model_name, response.model_version) != (
        request_id,
        target.name,
        target.version,
    ):
        raise ValueError("response request ID, model name or version mismatch")
    outputs = {o.name: o for o in response.outputs}
    if len(outputs) != len(response.outputs) or set(outputs) != {
        o.name for o in target.expected
    }:
        raise ValueError("response output names mismatch")
    wrong_values = []
    for expected in target.expected:
        actual = outputs[expected.name]
        if (actual.datatype, actual.shape) != (expected.datatype, expected.shape):
            raise ValueError(f"{expected.name}: datatype or shape mismatch")
        floating = expected.datatype in ("FP16", "FP32", "FP64")
        values = expected.data.root
        if isinstance(reply, bytes):
            if floating:
                allowed: tuple[type, ...] = (int, float)
            elif expected.datatype == "BOOL":
                allowed = (bool,)
            elif expected.datatype == "BYTES":
                allowed = (str,)
            else:
                allowed = (int,)
            if any(type(value) not in allowed for value in actual.data.root):
                raise ValueError(f"{expected.name}: contents disagree with datatype")
        elif expected.datatype == "BYTES":
            values = [encode_str(v) if isinstance(v, str) else v for v in values]
        # Type inference can round valid UINT64 integers to float64.
        observed = np.asarray(actual.data.root, dtype=None if floating else object)
        wanted = np.asarray(values, dtype=None if floating else object)
        if observed.shape != wanted.shape or observed.size != math.prod(expected.shape):
            raise ValueError(f"{expected.name}: data length mismatch")
        if floating:
            equal = np.allclose(observed, wanted, rtol=target.rtol, atol=target.atol)
        else:
            equal = np.array_equal(observed, wanted)
        if not equal:
            wrong_values.append(expected.name)
    if wrong_values:
        raise _IncorrectValues(f"{', '.join(wrong_values)}: incorrect values")


def database(path: Path) -> sqlite3.Connection:
    # Never append to a previous measurement.
    path.touch(exist_ok=False)
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE run(id TEXT, address TEXT, scenario TEXT);
        CREATE TABLE completed_run(id TEXT PRIMARY KEY);
        CREATE TABLE attempt(id TEXT PRIMARY KEY, phase TEXT, model TEXT, version TEXT,
            due REAL, submitted REAL, finished REAL, outcome TEXT, detail TEXT,
            verified_reply INTEGER NOT NULL);
        CREATE TABLE lifecycle(phase TEXT, model TEXT, operation TEXT, due REAL,
            submitted REAL, finished REAL, status TEXT, expected TEXT);
        CREATE TABLE resource(at REAL, source TEXT, subject TEXT, bytes INTEGER);
    """
    )
    return db


def _cgroup_memory_path() -> Path | None:
    cgroup = Path("/sys/fs/cgroup")
    membership = Path("/proc/self/cgroup")
    paths = []
    if membership.exists():
        for line in membership.read_text().splitlines():
            hierarchy, controllers, group = line.split(":", 2)
            relative = Path(group.lstrip("/"))
            if ".." not in relative.parts:
                if hierarchy == "0" and not controllers:
                    paths.append(cgroup / relative / "memory.current")
                elif "memory" in controllers.split(","):
                    paths.append(cgroup / "memory" / relative / "memory.usage_in_bytes")
    return next(
        (
            p
            for p in [
                *paths,
                cgroup / "memory.current",
                cgroup / "memory/memory.usage_in_bytes",
            ]
            if p.exists()
        ),
        None,
    )


async def _sample_gpu_memory(
    smi: str, db: sqlite3.Connection, now: Callable[[], float]
) -> None:
    process = await asyncio.create_subprocess_exec(
        smi,
        "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode:
        raise RuntimeError(f"nvidia-smi exited {process.returncode}: {stderr.decode()}")
    for line in stdout.decode().splitlines():
        try:
            gpu, mib = line.split(",")
            used = int(mib) * 1024**2
        except ValueError:
            logger.warning("Skipping unavailable GPU memory reading: %r", line)
            continue
        db.execute(
            "INSERT INTO resource VALUES(?,?,?,?)",
            (now(), "gpu_used", gpu.strip(), used),
        )


async def resources(
    db: sqlite3.Connection, now: Callable[[], float], stop: asyncio.Event
) -> None:
    """Sample the co-located Linux cgroup and visible GPUs off the inference path."""
    memory = None
    smi = shutil.which("nvidia-smi")
    while True:
        try:
            if memory is None:
                memory = _cgroup_memory_path()
            if memory is not None:
                db.execute(
                    "INSERT INTO resource VALUES(?,?,?,?)",
                    (now(), "cgroup_memory", str(memory), int(memory.read_text())),
                )
        except (OSError, ValueError) as exc:
            logger.warning("Skipping cgroup memory sample: %s", exc)
        if smi:
            try:
                await _sample_gpu_memory(smi, db, now)
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("Skipping GPU memory sample: %r", exc)
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except TimeoutError:
            pass


async def run(
    scenario: Scenario,
    client: GRPCInferenceServiceStub | aiohttp.ClientSession,
    db: sqlite3.Connection,
    address: str,
    local_resources: bool = False,
) -> None:
    clock = asyncio.get_running_loop().time
    run_id = uuid.uuid4().hex
    calls = (
        RestCalls(client, scenario.models)
        if isinstance(client, aiohttp.ClientSession)
        else GrpcCalls(client, scenario.models)
    )
    recorded = scenario.model_copy(
        update={"deployment": {**scenario.deployment, "protocol": calls.protocol}}
    )
    db.execute(
        "INSERT INTO run(id,address,scenario) VALUES(?,?,?)",
        (run_id, address, recorded.model_dump_json()),
    )
    db.commit()

    async def manage(action: Action, origin: float, phase: str) -> str:
        await asyncio.sleep(max(0, origin + action.at_s - clock()))
        submitted = clock() - origin
        try:
            await calls.manage(
                action.operation, action.model, scenario.management_timeout_s
            )
            status = "OK"
        except (grpc.aio.AioRpcError, aiohttp.ClientError, TimeoutError) as exc:
            _, status = failure(exc)
        db.execute(
            "INSERT INTO lifecycle VALUES(?,?,?,?,?,?,?,?)",
            (
                phase,
                action.model,
                action.operation,
                action.at_s,
                submitted,
                clock() - origin,
                status,
                action.expected_status,
            ),
        )
        return status

    async def traffic(duration: float, phase: str, origin: float) -> None:
        rng = random.Random(scenario.seed)
        active: set[asyncio.Task] = set()

        async def infer(index: int, request_id: str, due: float) -> None:
            target = scenario.models[index]
            submitted = clock() - origin
            detail = ""
            verified_reply = False
            try:
                reply = await calls.infer(index, request_id, scenario.deadline_s)
            except (grpc.aio.AioRpcError, aiohttp.ClientError, TimeoutError) as exc:
                finished = clock() - origin
                outcome, detail = failure(exc)
            else:
                finished = clock() - origin
                try:
                    check_output(reply, target, request_id)
                    outcome = "correct"
                    verified_reply = True
                except (TypeError, ValueError) as exc:
                    outcome, detail = "wrong_output", str(exc)
                    verified_reply = isinstance(exc, _IncorrectValues)
            db.execute(
                "INSERT INTO attempt VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    phase,
                    target.name,
                    target.version,
                    due,
                    submitted,
                    finished,
                    outcome,
                    detail,
                    verified_reply,
                ),
            )

        async with asyncio.TaskGroup() as tasks:
            for seq in count():
                due = seq / scenario.rate
                if due >= duration:
                    break
                await asyncio.sleep(max(0, origin + due - clock()))
                choices = [
                    i
                    for i, m in enumerate(scenario.models)
                    if m.start_s == 0 or (phase == "measure" and m.start_s <= due)
                ]
                index = rng.choices(
                    choices, weights=[scenario.models[i].weight for i in choices]
                )[0]
                target = scenario.models[index]
                rid = f"{run_id}.{phase}.{seq}"
                late = clock() - origin - due >= 1 / scenario.rate
                if late or len(active) >= scenario.concurrency:
                    db.execute(
                        "INSERT INTO attempt VALUES(?,?,?,?,?,?,?,?,?,0)",
                        (
                            rid,
                            phase,
                            target.name,
                            target.version,
                            due,
                            None,
                            None,
                            "generator_late" if late else "generator_full",
                            "",
                        ),
                    )
                    continue
                task = tasks.create_task(infer(index, rid, due))
                active.add(task)
                task.add_done_callback(active.discard)
            # Event-loop timers can wake just before the monotonic deadline.
            end = origin + duration
            while (remaining := end - clock()) > 0:
                await asyncio.sleep(remaining)

    async def changes(origin: float) -> None:
        for action in sorted(scenario.actions, key=lambda a: a.at_s):
            await manage(action, origin, "measure")

    try:
        for model in scenario.initial_load:
            if (
                await manage(
                    Action(at_s=0, operation="load", model=model), clock(), "setup"
                )
                != "OK"
            ):
                raise RuntimeError(f"initial load failed for {model}")
        await traffic(scenario.warmup_s, "warmup", clock())
        origin = clock()
        stop = asyncio.Event()
        async with asyncio.TaskGroup() as tasks:
            if local_resources:
                tasks.create_task(resources(db, lambda: clock() - origin, stop))
            try:
                async with asyncio.TaskGroup() as measured:
                    measured.create_task(
                        traffic(scenario.duration_s, "measure", origin)
                    )

                    measured.create_task(changes(origin))
            finally:
                stop.set()
        db.execute("INSERT INTO completed_run VALUES(?)", (run_id,))
    finally:
        # Keep completed observations when a run is interrupted.
        db.commit()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--protocol", choices=("grpc", "rest"), default="grpc")
    parser.add_argument("--address", help="host:port for gRPC or base URL for REST")
    parser.add_argument("--local-resources", action="store_true")
    args = parser.parse_args()
    scenario = Scenario.model_validate_json(args.scenario.read_text())
    db = database(args.database)
    try:
        if args.protocol == "rest":
            address = args.address or "http://localhost:8080"
            async with aiohttp.ClientSession(
                address, connector=aiohttp.TCPConnector(limit=0)
            ) as session:
                await run(scenario, session, db, address, args.local_resources)
        else:
            address = args.address or "localhost:8081"
            async with grpc.aio.insecure_channel(address) as channel:
                await run(
                    scenario,
                    GRPCInferenceServiceStub(channel),
                    db,
                    address,
                    args.local_resources,
                )
        result = summary(db)
        print(json.dumps(result, indent=2))
        unverified_targets = any(
            not model["windows"]["all"]["outcomes"].get("correct")
            for model in result["models"]
        )
        if (
            unverified_targets
            or db.execute(
                "SELECT 1 FROM attempt WHERE outcome='wrong_output' UNION ALL "
                "SELECT 1 FROM lifecycle WHERE status != expected LIMIT 1"
            ).fetchone()
        ):
            raise SystemExit(1)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
