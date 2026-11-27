import pytest
import asyncio
from unittest.mock import patch

from mlserver.errors import (
    ModelLoadError,
    ModelNotFound,
    ModelUnloadError,
    MLServerError,
)
from mlserver.registry import MultiModelRegistry
from mlserver.handlers import ModelRepositoryHandlers
from mlserver.settings import ModelSettings, ModelParameters
from mlserver.types import RepositoryIndexRequest, State

from ..fixtures import ErrorModel, SumModel


FAILED_LOAD_REASON = (
    "Model failed to load. See the load response and server logs for details."
)


async def test_index(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    sum_model_settings: ModelSettings,
):
    repo_index = list(await model_repository_handlers.index(repository_index_request))

    assert len(repo_index) == 1
    assert repo_index[0].name == sum_model_settings.name
    assert sum_model_settings.parameters is not None
    assert repo_index[0].version == sum_model_settings.parameters.version
    assert repo_index[0].state == State.READY


async def test_index_unavailable_model(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    sum_model_settings: ModelSettings,
):
    await model_repository_handlers.unload(sum_model_settings.name)
    repo_index = list(await model_repository_handlers.index(repository_index_request))

    assert len(repo_index) == 1
    assert repo_index[0].name == sum_model_settings.name
    assert sum_model_settings.parameters is not None
    assert repo_index[0].version == sum_model_settings.parameters.version
    assert repo_index[0].state == State.UNAVAILABLE
    assert repo_index[0].reason == ""


@pytest.mark.parametrize("version", ["2", None])
async def test_index_initial_load_failure(
    model_repository, sum_model_settings, load_error_model_settings, version, mocker
):
    registry = MultiModelRegistry()
    handlers = ModelRepositoryHandlers(model_repository, registry)
    candidate = load_error_model_settings
    candidate.name = sum_model_settings.name
    candidate.parameters.version = version
    catalog = [candidate]
    mocker.patch.object(model_repository, "list", return_value=catalog)
    request = RepositoryIndexRequest()

    with pytest.raises(MLServerError):
        await handlers.load(candidate.name)
    with pytest.raises(ModelNotFound):
        await handlers.unload(candidate.name)
    # A different version becomes the inference default, even when the failed
    # repository entry has no version of its own.
    default = await registry.load(sum_model_settings)
    assert await registry.get_model(candidate.name) is default
    (item,) = await handlers.index(request)
    assert item.state == State.UNAVAILABLE
    assert item.reason == FAILED_LOAD_REASON
    assert not (await handlers.index(RepositoryIndexRequest(ready=True))).root

    catalog.clear()
    assert not (await handlers.index(request)).root
    catalog.append(candidate)
    assert (await handlers.index(request))[0].reason == ""
    with pytest.raises(MLServerError):
        await handlers.load(candidate.name)
    working = sum_model_settings.model_copy(deep=True)
    working.parameters.version = version
    await registry.load(working)
    with pytest.raises(MLServerError):
        await handlers.load(candidate.name)
    (item,) = await handlers.index(request)
    assert (item.state, item.reason) == (State.READY, "")
    assert not registry.has_load_failure(candidate)
    await handlers.unload(candidate.name)
    assert (await handlers.index(request))[0].reason == ""


async def test_index_keeps_last_failure_during_retry(
    model_repository, load_error_model_settings, mocker
):
    settings = load_error_model_settings
    registry = MultiModelRegistry()
    handlers = ModelRepositoryHandlers(model_repository, registry)
    mocker.patch.object(model_repository, "list", return_value=[settings])
    request = RepositoryIndexRequest()
    with pytest.raises(MLServerError, match=ErrorModel.error_message):
        await handlers.load(settings.name)

    retry_error = "retry failed: " + "x" * 8192

    async def fail_retry(model):
        (row,) = await handlers.index(request)
        assert (row.state, row.reason) == (State.LOADING, "")
        assert registry.has_load_failure(settings)
        raise MLServerError(retry_error)

    mocker.patch.object(settings.implementation, "load", fail_retry)
    with pytest.raises(MLServerError) as failure:
        await handlers.load(settings.name)
    assert str(failure.value) == retry_error
    (row,) = await handlers.index(request)
    assert (row.state, row.reason) == (State.UNAVAILABLE, FAILED_LOAD_REASON)


async def test_successful_retry_preserves_other_models_failure(
    model_repository, load_error_model_settings, mocker
):
    first = load_error_model_settings
    first.parameters.version = "1"
    second = first.model_copy(deep=True, update={"name": "second"})
    mocker.patch.object(model_repository, "list", return_value=[first, second])
    handlers = ModelRepositoryHandlers(model_repository, MultiModelRegistry())
    for settings in (first, second):
        with pytest.raises(MLServerError, match=ErrorModel.error_message):
            await handlers.load(settings.name)

    first.parameters.load_error = False
    await handlers.load(first.name)
    rows = await handlers.index(RepositoryIndexRequest())
    assert {row.name: (row.state, row.reason) for row in rows} == {
        first.name: (State.READY, ""),
        second.name: (State.UNAVAILABLE, FAILED_LOAD_REASON),
    }


async def test_index_excludes_registry_only_models(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)

    try:
        repo_index = list(
            await model_repository_handlers.index(repository_index_request)
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert [item.name for item in repo_index] == [sum_model_settings.name]


async def test_index_includes_registry_only_models_when_requested(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)
    repository_index_request.include_registry_models = True

    try:
        repo_index = list(
            await model_repository_handlers.index(repository_index_request)
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert {item.name for item in repo_index} == {
        sum_model_settings.name,
        registry_only_settings.name,
    }


async def test_index_ready_filters_repository_and_registry_models(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    await model_repository_handlers.unload(sum_model_settings.name)

    registry_only_settings = sum_model_settings.copy(deep=True)
    registry_only_settings.name = "registry-only-model"
    await model_registry.load(registry_only_settings)

    repository_index_request.include_registry_models = True
    repository_index_request.ready = True
    try:
        repo_index = list(
            await model_repository_handlers.index(repository_index_request)
        )
    finally:
        await model_registry.unload(registry_only_settings.name)

    assert [item.name for item in repo_index] == [registry_only_settings.name]
    assert repo_index[0].state == State.READY


async def test_index_ready_excludes_not_ready_registry_models(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    not_ready_settings = sum_model_settings.copy(deep=True)
    not_ready_settings.name = "not-ready-model"
    await model_registry.load(not_ready_settings)
    not_ready_model = await model_registry.get_model(not_ready_settings.name)
    not_ready_model.ready = False

    try:
        repository_index_request.include_registry_models = True
        repository_index_request.ready = True

        repo_index = list(
            await model_repository_handlers.index(repository_index_request)
        )

        assert [item.name for item in repo_index] == [sum_model_settings.name]
        assert repo_index[0].state == State.READY
    finally:
        await model_registry.unload(not_ready_settings.name)


@pytest.mark.parametrize("ready,expected", [(None, 1), (True, 0), (False, 1)])
async def test_index_filter_ready(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    sum_model_settings: ModelSettings,
    ready: bool | None,
    expected: int,
):
    await model_repository_handlers.unload(sum_model_settings.name)

    repository_index_request.ready = ready
    repo_index = list(await model_repository_handlers.index(repository_index_request))

    assert len(repo_index) == expected


async def test_unload(
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    await model_repository_handlers.unload(sum_model_settings.name)

    with pytest.raises(ModelNotFound):
        await model_registry.get_model(sum_model_settings.name)


async def test_unload_preserves_sibling_load_failures(model_repository, mocker):
    registry = MultiModelRegistry()
    handlers = ModelRepositoryHandlers(model_repository, registry)
    catalog = [
        ModelSettings(
            name="A",
            implementation=ErrorModel,
            parameters=ModelParameters(
                version=version, load_error=version in ("3", "4")
            ),
        )
        for version in ("1", "2", "3", "4")
    ]
    mocker.patch.object(model_repository, "list", return_value=catalog)
    for settings in catalog[:2]:
        await registry.load(settings)
    for settings in catalog[2:]:
        with pytest.raises(MLServerError, match=ErrorModel.error_message):
            await registry.load(settings)

    await handlers.unload("A")

    rows = await handlers.index(RepositoryIndexRequest())
    assert {row.version: (row.state, row.reason) for row in rows} == {
        "1": (State.UNAVAILABLE, ""),
        "2": (State.UNAVAILABLE, ""),
        "3": (State.UNAVAILABLE, FAILED_LOAD_REASON),
        "4": (State.UNAVAILABLE, FAILED_LOAD_REASON),
    }


async def test_unload_not_found(
    model_repository_handlers: ModelRepositoryHandlers,
):
    with pytest.raises(ModelNotFound):
        await model_repository_handlers.unload("not-existing")


async def test_load_not_found(
    model_repository_handlers: ModelRepositoryHandlers,
):
    with pytest.raises(ModelNotFound):
        await model_repository_handlers.load("not-existing")


async def test_load_removes_stale_models(
    model_repository_handlers: ModelRepositoryHandlers,
    repository_index_request: RepositoryIndexRequest,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
):
    # Load a few models which are not present on the repository (including a
    # default one), therefore they will be stale
    stale_settings = sum_model_settings.copy(deep=True)
    assert stale_settings.parameters is not None
    stale_settings.parameters.version = None
    await model_registry.load(stale_settings)

    to_load = ["v0", "v1", "v2"]
    for version in to_load:
        stale_settings = sum_model_settings.copy(deep=True)
        assert stale_settings.parameters is not None
        stale_settings.parameters.version = version
        await model_registry.load(stale_settings)

    # Validate that the stale test models have been loaded
    registry_models = await model_registry.get_models(sum_model_settings.name)
    stale_length = (
        len(to_load)
        + 1  # Count the (stale) default model
        + 1  # Count the previous (non-stale) model
    )
    assert len(registry_models) == stale_length

    # Reload our model and validate whether stale models have been removed
    await model_repository_handlers.load(sum_model_settings.name)

    # Assert that stale models have been removed from both the registry (and
    # ensure they are not present on the repository either)
    registry_models = await model_registry.get_models(sum_model_settings.name)
    repo_models = list(await model_repository_handlers.index(repository_index_request))
    assert sum_model_settings.parameters is not None
    expected_version = sum_model_settings.parameters.version

    assert len(registry_models) == 1
    assert registry_models[0].version == expected_version

    assert len(repo_models) == 1
    assert repo_models[0].version == expected_version


async def test_load_stale_cleanup_attempts_all_with_failures(
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
    sum_model_settings: ModelSettings,
    mocker,
):
    """Test repository load cleanup uses return_exceptions
    and attempts all stale versions"""
    # Load 3 stale versions in registry
    stale_versions = ["v0", "v1", "v2"]
    for version in stale_versions:
        stale_settings = sum_model_settings.copy(deep=True)
        assert stale_settings.parameters is not None
        stale_settings.parameters.version = version
        await model_registry.load(stale_settings)

    # Mock v1 to fail unload
    models = await model_registry.get_models(sum_model_settings.name)
    v1_model = [m for m in models if m.version == "v1"][0]
    mocker.patch.object(v1_model, "unload", side_effect=MLServerError("Unload failed"))

    # Reload model from repository (v1.2.3) - should attempt
    # to cleanup all stale versions
    from mlserver.errors import ModelUnloadError

    with pytest.raises(ModelUnloadError, match="Failed to cleanup 1 of 3 stale"):
        await model_repository_handlers.load(sum_model_settings.name)

    # v1.2.3 should be loaded successfully
    loaded_model = await model_registry.get_model(sum_model_settings.name, "v1.2.3")
    assert loaded_model.ready

    # All stale models removed from registry (unregister happens before unload)
    all_models = await model_registry.get_models(sum_model_settings.name)
    assert len(all_models) == 1
    assert all_models[0].version == "v1.2.3"


async def test_load_serializes_same_model_operations(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: SumModel,
    sum_model_settings: ModelSettings,
    mocker,
):
    started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    async def find(_name: str):
        return [sum_model_settings]

    async def load(_settings):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        started.set()
        await release.wait()
        active -= 1
        return sum_model

    async def get_models(_name: str):
        return [sum_model]

    mocker.patch.object(model_repository_handlers._repository, "find", new=find)
    mocker.patch.object(model_repository_handlers._model_registry, "load", new=load)
    mocker.patch.object(
        model_repository_handlers._model_registry, "get_models", new=get_models
    )

    first = asyncio.create_task(model_repository_handlers.load(sum_model_settings.name))
    await started.wait()
    second = asyncio.create_task(
        model_repository_handlers.load(sum_model_settings.name)
    )
    await asyncio.sleep(0)

    assert not second.done()
    assert maximum_active == 1
    release.set()
    await asyncio.gather(first, second)


async def test_load_allows_different_model_operations_concurrently(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model_settings: ModelSettings,
    mocker,
):
    entered = 0
    maximum_active = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()
    active = 0

    first_settings = sum_model_settings.model_copy(deep=True)
    first_settings.name = "first-handler-model"
    second_settings = sum_model_settings.model_copy(deep=True)
    second_settings.name = "second-handler-model"
    settings_by_name = {
        first_settings.name: first_settings,
        second_settings.name: second_settings,
    }
    models_by_name = {
        name: SumModel(settings) for name, settings in settings_by_name.items()
    }

    async def find(name: str):
        return [settings_by_name[name]]

    async def load(settings: ModelSettings):
        nonlocal active, entered, maximum_active
        entered += 1
        active += 1
        maximum_active = max(maximum_active, active)
        if entered == 2:
            both_entered.set()
        await release.wait()
        active -= 1
        return models_by_name[settings.name]

    async def get_models(name: str):
        return [models_by_name[name]]

    mocker.patch.object(model_repository_handlers._repository, "find", new=find)
    mocker.patch.object(model_repository_handlers._model_registry, "load", new=load)
    mocker.patch.object(
        model_repository_handlers._model_registry, "get_models", new=get_models
    )

    first = asyncio.create_task(model_repository_handlers.load(first_settings.name))
    second = asyncio.create_task(model_repository_handlers.load(second_settings.name))
    await both_entered.wait()

    assert maximum_active == 2
    release.set()
    await asyncio.gather(first, second)


async def test_load_serializes_with_unload_for_same_model(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: SumModel,
    sum_model_settings: ModelSettings,
    mocker,
):
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    async def find(_name: str):
        return [sum_model_settings]

    async def load(_settings):
        load_started.set()
        await release_load.wait()
        return sum_model

    async def get_models(_name: str):
        return [sum_model]

    async def unload(_name: str):
        return None

    mocker.patch.object(model_repository_handlers._repository, "find", new=find)
    mocker.patch.object(model_repository_handlers._model_registry, "load", new=load)
    mocker.patch.object(
        model_repository_handlers._model_registry, "get_models", new=get_models
    )
    mocker.patch.object(model_repository_handlers._model_registry, "unload", new=unload)

    load_task = asyncio.create_task(
        model_repository_handlers.load(sum_model_settings.name)
    )
    await load_started.wait()
    unload_task = asyncio.create_task(
        model_repository_handlers.unload(sum_model_settings.name)
    )
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_load.set()
    await asyncio.gather(load_task, unload_task)


async def test_cancelled_load_waits_for_repository_operation_to_settle(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: SumModel,
    sum_model_settings: ModelSettings,
    mocker,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def find(_name: str):
        return [sum_model_settings]

    async def load(_settings):
        started.set()
        await release.wait()
        return sum_model

    async def get_models(_name: str):
        return [sum_model]

    mocker.patch.object(model_repository_handlers._repository, "find", new=find)
    mocker.patch.object(model_repository_handlers._model_registry, "load", new=load)
    mocker.patch.object(
        model_repository_handlers._model_registry, "get_models", new=get_models
    )

    task = asyncio.create_task(model_repository_handlers.load(sum_model_settings.name))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    release.set()
    await task

    # The per-model lock was released after deferred cancellation settled.
    await model_repository_handlers.load(sum_model_settings.name)


async def test_cancelled_unload_waits_for_repository_operation_to_settle(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model_settings: ModelSettings,
    mocker,
):
    unload_started = asyncio.Event()
    release_unload = asyncio.Event()

    async def unload(_name: str):
        unload_started.set()
        await release_unload.wait()

    mocker.patch.object(model_repository_handlers._model_registry, "unload", new=unload)

    unload_task = asyncio.create_task(
        model_repository_handlers.unload(sum_model_settings.name)
    )
    await unload_started.wait()
    unload_task.cancel()
    await asyncio.sleep(0)

    assert not unload_task.done()
    release_unload.set()
    await unload_task

    # The per-model lock was released after deferred cancellation settled.
    await model_repository_handlers.unload(sum_model_settings.name)


async def test_union_model_settings_deduplication_and_precedence(
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model_settings: ModelSettings,
):
    """Test _union_model_settings deduplicates by
    (name, version) and registry takes precedence"""
    from ..fixtures import SumModel

    repository_settings = [
        ModelSettings(
            name="model-a",
            implementation=SumModel,
            parameters=ModelParameters(version="v1", uri="repo-uri"),
        ),
        ModelSettings(
            name="model-b",
            implementation=SumModel,
            parameters=ModelParameters(version="v2"),
        ),
    ]

    registry_settings = [
        ModelSettings(
            name="model-a",
            implementation=SumModel,
            parameters=ModelParameters(version="v1", uri="registry-uri"),
        ),
        ModelSettings(
            name="model-c",
            implementation=SumModel,
            parameters=ModelParameters(version="v3"),
        ),
    ]

    result = model_repository_handlers._union_model_settings(
        repository_settings, registry_settings
    )

    # Should have 3 models: A v1, B v2, C v3
    assert len(result) == 3

    model_map = {
        (m.name, m.parameters.version if m.parameters else None): m for m in result
    }  # type: ignore

    # Model A v1 should use registry settings (not repository)
    assert model_map[("model-a", "v1")].parameters.uri == "registry-uri"  # type: ignore

    # Model B and C should be present
    assert ("model-b", "v2") in model_map
    assert ("model-c", "v3") in model_map


async def test_index_item_while_model_loads(
    model_repository_handlers: ModelRepositoryHandlers,
    model_registry: MultiModelRegistry,
):
    from ..fixtures import SlowModel

    # Load a slow model that takes time
    slow_settings = ModelSettings(name="slow-model", implementation=SlowModel)

    # Start loading in background
    load_task = asyncio.create_task(model_registry.load(slow_settings))
    await asyncio.sleep(0.1)  # Let it start loading

    # Query state while loading
    item = await model_repository_handlers._to_item(slow_settings)
    assert (item.state, item.reason) == (State.LOADING, "")

    # Wait for load to complete
    await load_task

    # Now should be READY
    item = await model_repository_handlers._to_item(slow_settings)
    assert (item.state, item.reason) == (State.READY, "")

    # Unload model
    await model_registry.unload("slow-model")

    # Should be UNAVAILABLE
    item = await model_repository_handlers._to_item(slow_settings)
    assert (item.state, item.reason) == (State.UNAVAILABLE, "")


@pytest.fixture
def abandoned_load(model_repository, sum_model_settings, mocker):
    """Hold a SumModel load past its deadline and release it on demand."""
    registry = MultiModelRegistry(on_model_load=[], model_operation_timeout=5)
    handlers = ModelRepositoryHandlers(model_repository, registry)
    catalog = [sum_model_settings]
    mocker.patch.object(model_repository, "list", return_value=catalog)

    async def index():
        return {
            (item.name, item.version): (item.state, item.reason)
            for item in await handlers.index(RepositoryIndexRequest())
        }

    async def start(outcome, *, cleanup_error=None):
        entered = asyncio.get_running_loop().create_future()
        release = asyncio.Event()

        async def prepare(model):
            entered.set_result(model)
            await release.wait()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        registry._model_operation_timeout = 0.1
        with patch.object(sum_model_settings.implementation, "load", prepare):
            operation = asyncio.create_task(
                registry.load(sum_model_settings.model_copy(deep=True))
            )
            try:
                candidate = await entered
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.shield(operation)
            except BaseException:
                release.set()
                raise
        if cleanup_error:
            mocker.patch.object(candidate, "unload", side_effect=cleanup_error)
        single = registry._models[candidate.name]
        registry._model_operation_timeout = 5
        return candidate, release, single._cleanup_tasks[candidate]

    return registry, catalog, index, start


@pytest.mark.parametrize("outcome", [ValueError("late preparation error"), False, True])
async def test_index_after_abandoned_preparation(
    abandoned_load, outcome, caplog, mocker
):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    key = requested.name, requested.version
    sibling = requested.model_copy(deep=True)
    sibling.parameters.version = "sibling"
    catalog.append(sibling)
    async with asyncio.timeout(10):
        await registry.load(sibling)
        candidate, release, cleanup = await start(outcome)
        try:
            assert not cleanup.done()
            assert (await index())[key] == (State.UNAVAILABLE, "")
            assert not registry.has_load_failure(requested)
        finally:
            release.set()
        await asyncio.shield(cleanup)
        failed = outcome is not True
        assert (await index()) == {
            key: (State.UNAVAILABLE, FAILED_LOAD_REASON if failed else ""),
            (sibling.name, sibling.version): (State.READY, ""),
        }
        with pytest.raises(ModelNotFound):
            await registry.get_model(*key)
        if failed:
            assert (
                "returned False" if outcome is False else str(outcome)
            ) in caplog.text

        async def prepare_retry(model):
            assert (await index())[key] == (State.LOADING, "")
            assert registry.has_load_failure(requested) is failed
            return True

        mocker.patch.object(requested.implementation, "load", prepare_retry)
        current = await registry.load(requested)
        assert current is not candidate
        assert await registry.get_model(*key) is current
        assert (await index())[key] == (State.READY, "")
        assert not registry.has_load_failure(requested)


@pytest.mark.parametrize("previous_failure", [False, True])
async def test_catalog_removal_invalidates_abandoned_load(
    abandoned_load, previous_failure
):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    key = requested.name, requested.version
    async with asyncio.timeout(10):
        if previous_failure:
            with patch.object(requested.implementation, "load", return_value=False):
                with pytest.raises(ModelLoadError, match="returned False"):
                    await registry.load(requested)
            assert (await index())[key] == (State.UNAVAILABLE, FAILED_LOAD_REASON)
        _, release, cleanup = await start(ValueError("obsolete load"))
        try:
            catalog.clear()
            assert await index() == {}
            catalog.append(requested)
        finally:
            release.set()
        await asyncio.shield(cleanup)
        assert (await index())[key] == (State.UNAVAILABLE, "")
        assert not registry.has_load_failure(requested)


async def test_abandoned_load_keeps_preparation_and_cleanup_failures(
    abandoned_load, caplog, mocker
):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    primary = ValueError("late preparation error")
    secondary = RuntimeError("resource still held")
    async with asyncio.timeout(10):
        candidate, release, cleanup = await start(primary, cleanup_error=secondary)
        release.set()
        with pytest.raises(RuntimeError) as error:
            await asyncio.shield(cleanup)
        assert error.value is secondary
        assert {record.exc_info[1] for record in caplog.records if record.exc_info} >= {
            primary,
            secondary,
        }
        assert (await index())[(requested.name, requested.version)] == (
            State.UNAVAILABLE,
            FAILED_LOAD_REASON,
        )
        with pytest.raises(RuntimeError) as retry:
            await registry.load(requested)
        assert retry.value is secondary
        mocker.stop(candidate.unload)
        await registry.unload(requested.name)
        assert registry.has_load_failure(requested)


async def test_abandoned_replacement_failure_leaves_original_ready(abandoned_load):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    key = requested.name, requested.version
    async with asyncio.timeout(10):
        original = await registry.load(requested)
        _, release, cleanup = await start(ValueError("failed replacement"))
        release.set()
        await asyncio.shield(cleanup)
        assert await registry.get_model(*key) is original
        assert (await index())[key] == (State.READY, "")
        assert not registry.has_load_failure(requested)


async def test_cancelled_preparation_keeps_uncertain_reason(abandoned_load):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    async with asyncio.timeout(10):
        _, release, cleanup = await start(asyncio.CancelledError())
        release.set()
        with pytest.raises(ModelUnloadError, match="ownership is retained"):
            await asyncio.shield(cleanup)
        assert (await index())[(requested.name, requested.version)] == (
            State.UNAVAILABLE,
            "",
        )
        assert not registry.has_load_failure(requested)


async def test_abandoned_failure_uses_requested_version(abandoned_load):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    redirected = requested.model_copy(deep=True)
    redirected.parameters.version = "redirected"
    catalog.append(redirected)

    async def change_version(model):
        model.settings.parameters.version = redirected.version
        return model

    registry._on_model_load = [change_version]
    async with asyncio.timeout(10):
        candidate, release, cleanup = await start(False)
        release.set()
        await asyncio.shield(cleanup)
        assert candidate.version == redirected.version
        assert await index() == {
            (requested.name, requested.version): (
                State.UNAVAILABLE,
                FAILED_LOAD_REASON,
            ),
            (redirected.name, redirected.version): (State.UNAVAILABLE, ""),
        }


async def test_redirected_retry_clears_requested_failure(abandoned_load, mocker):
    registry, catalog, index, start = abandoned_load
    requested = catalog[0]
    requested.parameters.version = "1"
    redirected = requested.model_copy(deep=True)
    redirected.parameters.version = "2"
    catalog.append(redirected)

    _, release, cleanup = await start(False)
    release.set()
    await asyncio.shield(cleanup)
    assert registry.has_load_failure(requested)

    async def change_version(model):
        model.settings.parameters.version = redirected.version
        return model

    registry._on_model_load.append(change_version)
    try:
        mocker.patch.object(requested.implementation, "load", return_value=True)

        current = await registry.load(requested.model_copy(deep=True))

        assert current.version == redirected.version
        assert not registry.has_load_failure(requested)
        assert await index() == {
            (requested.name, requested.version): (State.UNAVAILABLE, ""),
            (redirected.name, redirected.version): (State.READY, ""),
        }
    finally:
        registry._on_model_load.remove(change_version)
        await registry.unload(requested.name)
