import asyncio
from copy import deepcopy
from pathlib import Path
import os

import pytest

from mlserver.handlers.dataplane import DataPlane
from mlserver.errors import ModelNotFound
from mlserver.handlers.model_repository import ModelRepositoryHandlers
from mlserver.parallel.errors import NoWorkersAvailable
from mlserver.parallel.dispatcher import DispatchUpdateSettlement
from mlserver.parallel.pool import InferencePool
from mlserver.parallel.messages import ModelRequestMessage
from mlserver.registry import MultiModelRegistry
from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import RepositoryIndexRequest, State
from tests.fixtures import ErrorModel


async def test_timeout_quarantines_until_rollback(
    inference_pool, sum_model, monkeypatch
):
    # Purpose: Exercise frontend timeout while a delivered worker update remains
    # unsettled.
    # Expect: The candidate is excluded from replay and workers are unroutable until
    # owned recovery permits retry.
    # Method: Replace update dispatch with an event-held acknowledgement, await the
    # deadline, check routing rejection, then release and retry the load.
    entered, release = asyncio.Event(), asyncio.Event()
    pool = inference_pool
    pool._settings.model_operation_timeout = 0.05

    async def dispatch(message):
        entered.set()
        await release.wait()
        return DispatchUpdateSettlement([])

    monkeypatch.setattr(pool._dispatcher, "dispatch_update", dispatch)
    operation = asyncio.create_task(pool.load_model(sum_model))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert not pool.has_model(sum_model.settings)
        with pytest.raises(NoWorkersAvailable):
            pool._dispatcher._get_worker()
        release.set()
        pool._settings.model_operation_timeout = 5
        retry = await asyncio.wait_for(pool.load_model(sum_model), 10)
        assert retry is not None
        assert pool.has_model(sum_model.settings)
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


async def controlled_worker_load(model):
    """Pause before the runtime's load method using files visible across spawn."""
    directory = model.settings.parameters.extra.get("lifecycle_barrier")
    if directory:
        path = Path(directory)
        (path / f"entered-{os.getpid()}").touch()
        while not (path / f"release-{os.getpid()}").exists():
            await asyncio.sleep(0.01)
        if (path / f"fail-{os.getpid()}").exists():
            raise ValueError("late worker failure")
    return model


async def wait_until(predicate):
    while not predicate():
        await asyncio.sleep(0.01)


@pytest.mark.parametrize("transport", ["queue", "zmq"])
async def test_only_registry_hook_calls_stage_pool_loads(
    settings, sum_model, transport
):
    if transport == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_inference_transport = transport
    settings.parallel_workers = 1
    pool = InferencePool(settings)
    managed = sum_model.settings.model_copy(deep=True, update={"name": "managed"})

    async def check_staged(model):
        assert not pool.has_model(model.settings)
        return model

    class Hooks(list):
        def __iter__(self):
            # This independent task inherits the context used to enumerate hooks.
            tasks.create_task(pool.load_model(sum_model))
            return super().__iter__()

    registry = MultiModelRegistry(on_model_load=Hooks([pool.load_model, check_staged]))
    async with asyncio.timeout(30):
        await pool.start()
        try:
            async with asyncio.TaskGroup() as tasks:
                await registry.load(managed)
            assert pool.has_model(managed)
            assert pool.has_model(sum_model.settings)
            assert not pool._staged
        finally:
            await pool.close()


@pytest.mark.parametrize(
    "delegation",
    ["await", "handled", "handled-same-candidate", "completed-hook", "outlives-hook"],
)
async def test_load_hook_attributes_only_its_own_worker_failure(
    settings, sum_model, load_error_model, tmp_path, delegation
):
    settings.parallel_workers = 1
    settings.model_operation_timeout = 15
    pool = InferencePool(settings, on_worker_load=[controlled_worker_load])
    load_error_model.settings.parameters.extra["lifecycle_barrier"] = str(tmp_path)
    waiting = asyncio.Event()
    target = load_error_model.settings.model_copy(deep=True)
    target.parameters.version = "requested"
    worker_load = None

    async with asyncio.timeout(30):
        await pool.start()
        (pid,) = pool._workers
        release = tmp_path / f"release-{pid}"
        try:

            async def cancel_worker_load():
                assert worker_load is not None
                worker_load.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker_load

            async def first_hook(model):
                nonlocal worker_load
                if delegation == "await":
                    # The hook delegates its own preparation to a different instance.
                    return await pool.load_model(load_error_model)
                candidate = (
                    model
                    if delegation == "handled-same-candidate"
                    else load_error_model
                )
                worker_load = asyncio.create_task(pool.load_model(candidate))
                await wait_until((tmp_path / f"entered-{pid}").exists)
                if delegation == "outlives-hook":
                    return model
                await cancel_worker_load()
                if delegation == "completed-hook":
                    return model
                return await later_hook(model)

            async def later_hook(model):
                if delegation == "outlives-hook":
                    await cancel_worker_load()
                waiting.set()
                await asyncio.Event().wait()

            registry = MultiModelRegistry(
                on_model_load=[first_hook, later_hook],
                model_operation_timeout=1,
            )
            # Warm the worker before applying the frontend deadline.
            await pool.load_model(sum_model)
            with pytest.raises(asyncio.TimeoutError):
                await registry.load(target)
            assert (tmp_path / f"entered-{pid}").exists()
            assert waiting.is_set() is (delegation != "await")
            assert not registry.has_load_failure(target)
            release.touch()
            await pool._settle_recovery()
            assert registry.has_load_failure(target) is (delegation == "await")
            assert not registry.has_load_failure(load_error_model.settings)
        finally:
            release.touch()
            await pool.close()


async def test_cancelled_task_group_reports_already_settled_worker_load(
    settings, sum_model, load_error_model, tmp_path
):
    settings.parallel_workers = 1
    settings.model_operation_timeout = 15
    pool = InferencePool(settings, on_worker_load=[controlled_worker_load])
    candidate = load_error_model.settings
    candidate.parameters.extra["lifecycle_barrier"] = str(tmp_path)

    async def load_in_group(model):
        try:
            async with asyncio.TaskGroup() as group:
                load = group.create_task(pool.load_model(model))
        except asyncio.CancelledError:
            # Let the worker fail and rollback finish before the task group
            # passes its own cancellation back to the registry.
            assert entered.exists()
            release.touch()
            await pool._settle_recovery()
            assert not registry.has_load_failure(candidate)
            raise
        return load.result()

    registry = MultiModelRegistry(
        on_model_load=[load_in_group], on_model_unload=[pool.unload_model]
    )
    async with asyncio.timeout(30):
        await pool.start()
        (pid,) = pool._workers
        entered = tmp_path / f"entered-{pid}"
        release = tmp_path / f"release-{pid}"
        try:
            await registry.load(sum_model.settings)
            registry._model_operation_timeout = 1
            with pytest.raises(asyncio.TimeoutError):
                await registry.load(candidate)
            await registry._models[candidate.name]._drain_cleanup()
            assert registry.has_load_failure(candidate)
            assert not registry.has_load_failure(sum_model.settings)
            assert not pool.has_model(candidate)
        finally:
            release.touch()
            await pool.close()


@pytest.mark.parametrize("transport", ["queue", "zmq"])
@pytest.mark.parametrize(
    "outcome",
    ["exception", "runtime-timeout", "false", "success", "removed", "worker-timeout"],
)
async def test_abandoned_worker_load_index_reason(
    settings,
    sum_model_settings,
    model_repository,
    inference_request,
    tmp_path,
    mocker,
    caplog,
    transport,
    outcome,
):
    if transport == "zmq":
        pytest.importorskip("zmq")
    settings.parallel_workers = 1
    settings.parallel_inference_transport = transport
    settings.model_operation_timeout = 5 if outcome == "worker-timeout" else 15
    pool = InferencePool(settings, on_worker_load=[controlled_worker_load])
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=15,
    )
    candidate = ModelSettings(
        name=sum_model_settings.name,
        implementation=ErrorModel,
        parameters=ModelParameters(
            version="candidate",
            extra={"lifecycle_barrier": str(tmp_path)},
            load_error=outcome in ("exception", "removed"),
            load_timeout=outcome == "runtime-timeout",
            load_returns_false=outcome == "false",
        ),
    )
    catalog = [sum_model_settings, candidate]
    mocker.patch.object(model_repository, "list", return_value=catalog)
    repository = ModelRepositoryHandlers(model_repository, registry)

    async def index():
        return {
            item.version: (item.state, item.reason)
            for item in await repository.index(RepositoryIndexRequest())
        }

    async with asyncio.timeout(45):
        await pool.start()
        (pid,) = pool._workers
        release = tmp_path / f"release-{pid}"
        operation = None
        try:
            original = await registry.load(sum_model_settings)
            registry._model_operation_timeout = 0.5
            operation = asyncio.create_task(registry.load(candidate))
            await asyncio.wait_for(wait_until((tmp_path / f"entered-{pid}").exists), 5)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.shield(operation)
            assert (await index())[candidate.version] == (State.UNAVAILABLE, "")
            assert pool._recovery_task is not None
            single = registry._models[candidate.name]
            (cleanup,) = single._cleanup_tasks.values()
            if outcome == "removed":
                catalog.remove(candidate)
                assert candidate.version not in await index()
                catalog.append(candidate)
            if outcome != "worker-timeout":
                release.touch()
            # Recovery joins the Load reply and the rollback acknowledgement.
            await pool._settle_recovery()
            await asyncio.shield(cleanup)

            reason = ""
            if outcome in ("exception", "runtime-timeout", "false"):
                reason = (
                    "Model failed to load. See the load response and server logs "
                    "for details."
                )
            assert await index() == {
                candidate.version: (State.UNAVAILABLE, reason),
                original.version: (State.READY, ""),
            }
            if outcome in ("exception", "removed"):
                assert ErrorModel.error_message in caplog.text
            elif outcome == "false":
                assert "returned False" in caplog.text
            elif outcome == "runtime-timeout":
                assert (
                    "builtins.TimeoutError: runtime artifact read timed out"
                    in caplog.text
                )
            with pytest.raises(ModelNotFound):
                await registry.get_model(candidate.name, candidate.version)
            assert not pool.has_model(candidate)
            assert await registry.get_model(original.name, original.version) is original
            response = await original.predict(inference_request)
            assert response.outputs[0].data.root == [6]
            if reason:
                retry = sum_model_settings.model_copy(deep=True)
                retry.parameters.version = candidate.version
                registry._model_operation_timeout = 15
                await registry.load(retry)
                assert (await index())[candidate.version] == (State.READY, "")
                assert not registry.has_load_failure(candidate)
        finally:
            release.touch()
            if operation is not None:
                await asyncio.gather(operation, return_exceptions=True)
            await pool.close()


@pytest.mark.parametrize("outcome", ["success", "failure", "death"])
@pytest.mark.usefixtures("prometheus_registry")
async def test_multiprocess_partial_load_recovery(
    settings, sum_model_settings, inference_request, model_repository, tmp_path, outcome
):
    # Purpose: Exercise a replacement timeout after one of two real workers acknowledges
    # while the other later succeeds, fails or dies.
    # Expect: Committed identity/replay survive, the shared pool is quarantined while
    # uncertain, and recovery/retry restores correct actual serving.
    # Method: Use per-process file barriers to release acknowledgements separately, then
    # query prediction and worker metadata after settlement and replacement.
    # Worker deadlines are generous. Only the frontend deadline expires, after
    # both worker hooks have entered and one worker has acknowledged its load.
    settings.model_operation_timeout = 10
    pool = InferencePool(settings, on_worker_load=[controlled_worker_load])
    await pool.start()
    original_workers = list(pool._workers.values())
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=10,
    )
    probes = DataPlane(settings, registry)
    probes.startup_complete()
    repository = ModelRepositoryHandlers(model_repository, registry)
    ready_only = RepositoryIndexRequest(ready=True, include_registry_models=True)
    replacement = deepcopy(sum_model_settings)
    replacement.platform = "abandoned-platform"
    replacement.parameters.extra["lifecycle_barrier"] = str(tmp_path)
    operation = None
    retry = None
    try:
        old = await registry.load(sum_model_settings)
        unrelated_settings = deepcopy(sum_model_settings)
        unrelated_settings.name = "unrelated-model"
        unrelated = await registry.load(unrelated_settings)
        registry._model_operation_timeout = 0.5
        operation = asyncio.create_task(registry.load(replacement))
        await asyncio.wait_for(
            wait_until(lambda: len(list(tmp_path.glob("entered-*"))) == 2), 5
        )
        first, second = original_workers
        (tmp_path / f"release-{first.pid}").touch()
        await asyncio.wait_for(
            wait_until(lambda: len(pool._dispatcher._async_responses._futures) == 1), 5
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert await registry.get_model(old.name, old.version) is old
        assert old.ready
        with pytest.raises(NoWorkersAvailable):
            pool._dispatcher._get_worker()
        with pytest.raises(NoWorkersAvailable):
            await unrelated.predict(inference_request)
        for strict in (True, False):
            settings.strict_readiness = strict
            for _ in range(2):
                assert await probes.ready() is (not strict)
                assert not await probes.model_ready(old.name, old.version)
                assert not await probes.model_ready(unrelated.name, unrelated.version)
            for model in (old, unrelated):
                item = await repository._to_item(model.settings)
                assert item.state == State.UNAVAILABLE
                assert item.reason == "Model is loaded but has no eligible worker."
            assert not (await repository.index(ready_only)).root
        assert list(pool._worker_registry.models) == [old.settings, unrelated.settings]
        assert not pool._pending_reload
        registry._model_operation_timeout = 10
        retry = asyncio.create_task(registry.load(sum_model_settings))
        await asyncio.sleep(0)
        assert not retry.done()
        if outcome == "failure":
            (tmp_path / f"fail-{second.pid}").touch()
        if outcome == "death":
            second.kill()
            await asyncio.to_thread(second.join, 5)
            await pool.on_worker_stop(second.pid, -9)
        else:
            (tmp_path / f"release-{second.pid}").touch()
        current = await asyncio.wait_for(retry, 15)
        assert current.ready
        assert current is not old
        settings.strict_readiness = True
        assert await probes.ready()
        assert await probes.model_ready(current.name, current.version)
        assert await probes.model_ready(unrelated.name, unrelated.version)
        assert {row.name: row.reason for row in await repository.index(ready_only)} == {
            current.name: "",
            unrelated.name: "",
        }
        response = await asyncio.wait_for(current.predict(inference_request), 5)
        assert response.model_name == old.name
        assert len(response.outputs) == 1
        unaffected = await unrelated.predict(inference_request)
        assert unaffected.model_name == unrelated.name
        assert await registry.get_model(unrelated.name, unrelated.version) is unrelated
        if outcome == "death":
            await asyncio.wait_for(
                wait_until(lambda: len(pool._dispatcher._ready_workers) == 2), 10
            )
        for _ in range(2):
            metadata = await current._send("metadata")
            assert metadata.platform == sum_model_settings.platform
    finally:
        for worker in original_workers:
            (tmp_path / f"release-{worker.pid}").touch()
        pending = [task for task in (operation, retry) if task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await pool.close()
        assert all(not worker.is_alive() for worker in original_workers)


async def test_later_hook_timeout_cannot_enter_replay(
    inference_pool, sum_model_settings
):
    # Purpose: Exercise a frontend hook deadline after workers have prepared a
    # replacement.
    # Expect: The old instance/replay settings survive, abandoned staging is removed and
    # acknowledged recovery restores routing.
    # Method: Pause the later hook with events, inspect replay before timeout, then
    # await recovery and explicitly unload the committed model.
    pool = inference_pool
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=10,
    )
    old = await registry.load(sum_model_settings)
    entered, release = asyncio.Event(), asyncio.Event()

    async def later_hook(model):
        entered.set()
        await release.wait()
        return model

    registry._models[old.name]._on_model_load = [pool.load_model, later_hook]
    registry._model_operation_timeout = 0.1
    operation = asyncio.create_task(registry.load(sum_model_settings))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert list(pool._worker_registry.models) == [old.settings]
        assert not pool._pending_reload
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(operation), 5)
        assert await registry.get_model(old.name, old.version) is old
        assert not pool._staged
        assert not pool._pending_reload
        await asyncio.wait_for(pool._settle_recovery(), 5)
        assert len(pool._dispatcher._ready_workers) == 2
        await registry.unload(old.name)
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


async def test_environment_extraction_retains_ownership(
    inference_pool_registry, monkeypatch
):
    # Purpose: Exercise a cancelled extraction waiter followed by another request for
    # the same environment.
    # Expect: The second request joins the original extraction rather than reusing
    # partial work or starting a duplicate.
    # Method: Hold a controlled from_tarball coroutine with events, cancel the first
    # waiter and count calls when the second completes.
    from mlserver.env import Environment

    registry = inference_pool_registry
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def extract(tarball, env_path, env_hash):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return Environment(env_path, env_hash)

    monkeypatch.setattr(Environment, "from_tarball", extract)
    first = asyncio.create_task(registry._extract_tarball("owned", "unused"))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(registry._extract_tarball("owned", "unused"))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        env = await asyncio.wait_for(second, 5)
        assert env.env_hash == "owned"
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )


async def test_incomplete_extraction_is_not_reused_after_owner_exit(
    inference_pool_registry,
):
    # Purpose: Check persistent incomplete-extraction detection after the original task
    # owner is gone.
    # Expect: A directory carrying the incomplete marker must be rejected rather than
    # treated as an activated environment.
    # Method: Create that marker at the registry's environment path and call extraction
    # through a new request, expecting an explicit failure.
    registry = inference_pool_registry
    directory = Path(registry._get_env_path("incomplete"))
    directory.mkdir()
    (directory / ".mlserver-extraction-incomplete").touch()
    with pytest.raises(RuntimeError, match="incomplete extraction"):
        await registry._extract_tarball("incomplete", "unused")


async def assert_worker_models(pool, model):
    # Address every actual process explicitly; frontend metadata is cached and
    # round-robin sampling alone would not establish which worker was checked.
    for worker in list(pool._workers.values()):
        request = ModelRequestMessage(
            model_name=model.name, model_version=model.version, method_name="metadata"
        )
        worker.send_request(request)
        response = await asyncio.wait_for(
            pool._dispatcher._async_responses.schedule_and_wait(request, worker), 5
        )
        metadata = response.return_value
        assert metadata.name == model.name
        assert metadata.versions == model.settings.versions
        assert metadata.platform == model.settings.platform


@pytest.mark.parametrize(
    "old_version,failed_version",
    [("1", None), ("1", "2"), (None, "2"), (None, None), ("1", "1")],
)
async def test_failed_load_preserves_every_committed_worker(
    settings, sum_model_settings, old_version, failed_version
):
    # Purpose: Exercise ordinary load failure across versioned/unversioned and same-
    # version identity combinations.
    # Expect: Compensating cleanup must preserve the exact committed model and an
    # unrelated model on every worker, with routing restored.
    # Method: Load trusted models on two real workers, fail ErrorModel preparation,
    # await recovery and address each worker directly for metadata.
    settings.model_operation_timeout = 10
    pool = InferencePool(settings)
    await pool.start()
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=10,
    )
    committed = sum_model_settings.model_copy(deep=True)
    committed.parameters.version = old_version
    try:
        old = await registry.load(committed)
        unrelated_settings = committed.model_copy(deep=True)
        unrelated_settings.name = "unrelated-version-control"
        unrelated = await registry.load(unrelated_settings)
        failing = ModelSettings(
            name=old.name,
            implementation=ErrorModel,
            parameters=ModelParameters(version=failed_version, load_error=True),
        )
        with pytest.raises(Exception):
            await registry.load(failing)
        await asyncio.wait_for(pool._settle_recovery(), 10)
        assert await registry.get_model(old.name, old.version) is old
        assert old.ready
        assert pool.has_model(old.settings)
        assert len(pool._dispatcher._ready_workers) == 2
        await assert_worker_models(pool, old)
        await assert_worker_models(pool, unrelated)
    finally:
        await pool.close()


@pytest.mark.parametrize("outcome", ["error", "late-success", "late-failure"])
async def test_unversioned_partial_load_preserves_versioned_workers(
    settings, sum_model_settings, tmp_path, outcome
):
    # Purpose: Exercise partial unversioned preparation with ordinary error or late
    # success/failure after timeout.
    # Expect: Exact rollback cannot fall through to the healthy versioned default; every
    # worker retains committed configuration and accepts a later retry.
    # Method: Use two real workers and per-PID file barriers, release one
    # acknowledgement first, then settle the other and inspect each process explicitly.
    settings.model_operation_timeout = 10
    pool = InferencePool(settings, on_worker_load=[controlled_worker_load])
    await pool.start()
    registry = MultiModelRegistry(
        on_model_load=[pool.load_model],
        on_model_unload=[pool.unload_model],
        model_operation_timeout=10,
    )
    workers = list(pool._workers.values())
    attempt = None
    try:
        committed = sum_model_settings.model_copy(deep=True)
        committed.parameters.version = "1"
        old = await registry.load(committed)
        unrelated_settings = committed.model_copy(deep=True)
        unrelated_settings.name = "unrelated-partial-control"
        unrelated = await registry.load(unrelated_settings)
        incoming = committed.model_copy(deep=True)
        incoming.parameters.version = None
        incoming.platform = "abandoned-unversioned-platform"
        incoming.parameters.extra["lifecycle_barrier"] = str(tmp_path)
        if outcome != "error":
            registry._model_operation_timeout = 0.5
        attempt = asyncio.create_task(registry.load(incoming))
        await asyncio.wait_for(
            wait_until(lambda: len(list(tmp_path.glob("entered-*"))) == 2), 5
        )
        first, second = workers
        (tmp_path / f"release-{first.pid}").touch()
        await asyncio.wait_for(
            wait_until(lambda: len(pool._dispatcher._async_responses._futures) == 1), 5
        )
        if outcome != "error":
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(attempt), 5)
            with pytest.raises(NoWorkersAvailable):
                pool._dispatcher._get_worker()
        if outcome != "late-success":
            (tmp_path / f"fail-{second.pid}").touch()
        (tmp_path / f"release-{second.pid}").touch()
        if outcome == "error":
            with pytest.raises(Exception) as error:
                await asyncio.wait_for(asyncio.shield(attempt), 10)
            assert not isinstance(error.value, asyncio.TimeoutError)
        await asyncio.wait_for(pool._settle_recovery(), 10)
        assert await registry.get_model(old.name) is old
        assert old.ready
        assert list(pool._worker_registry.models) == [old.settings, unrelated.settings]
        assert len(pool._dispatcher._ready_workers) == 2
        await assert_worker_models(pool, old)
        await assert_worker_models(pool, unrelated)
        registry._model_operation_timeout = 10
        current = await registry.load(committed.model_copy(deep=True))
        assert current.ready
        await assert_worker_models(pool, current)
    finally:
        for worker in workers:
            (tmp_path / f"release-{worker.pid}").touch()
        if attempt is not None:
            await asyncio.gather(attempt, return_exceptions=True)
        await pool.close()
        assert all(not worker.is_alive() for worker in workers)
