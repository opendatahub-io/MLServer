import pytest
import asyncio
import platform

from unittest.mock import patch

from mlserver.utils import (
    defer_cancellation,
    get_model_uri,
    extract_headers,
    insert_headers,
    get_normalized_version,
    install_uvloop_event_loop,
    with_operation_lock,
)
from mlserver.version import __version__
from mlserver.types import InferenceRequest, InferenceResponse, Parameters
from mlserver.settings import ModelSettings, ModelParameters
from .fixtures import SumModel

test_get_model_uri_paramaters = [
    ("s3://bucket/key", None, "s3://bucket/key"),
    ("s3://bucket/key", "/mnt/models/model-settings.json", "s3://bucket/key"),
]
for scheme in ["", "file:"]:
    for uri, source, expected in [
        ("my-model.bin", None, "my-model.bin"),
        (
            "my-model.bin",
            "./my-model-folder/model-settings.json",
            "my-model-folder/my-model.bin",
        ),
        (
            "my-model.bin",
            "./my-model-folder/../model-settings.json",
            "my-model.bin",
        ),
        (
            "/an/absolute/path/my-model.bin",
            "/mnt/models/model-settings.json",
            "/an/absolute/path/my-model.bin",
        ),
    ]:
        test_get_model_uri_paramaters.append((scheme + uri, source, expected))


@pytest.mark.parametrize(
    "uri, source, expected",
    test_get_model_uri_paramaters,
)
async def test_get_model_uri(uri: str, source: str | None, expected: str):
    model_settings = ModelSettings(
        implementation=SumModel, parameters=ModelParameters(uri=uri)
    )
    model_settings._source = source
    with patch("os.path.isfile", return_value=True):
        model_uri = await get_model_uri(model_settings)

    assert model_uri == expected


@pytest.mark.parametrize(
    "parameters",
    [
        None,
        Parameters(),
        Parameters(headers={"foo": "bar2"}),
        Parameters(headers={"bar": "foo"}),
    ],
)
def test_insert_headers(parameters: Parameters):
    inference_request = InferenceRequest(inputs=[], parameters=parameters)
    headers = {"foo": "bar", "hello": "world"}
    insert_headers(inference_request, headers)

    assert inference_request.parameters is not None
    assert inference_request.parameters.headers == headers


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (None, None),
        (Parameters(), None),
        (Parameters(headers={}), {}),
        (Parameters(headers={"foo": "bar"}), {"foo": "bar"}),
    ],
)
def test_extract_headers(parameters: Parameters, expected: dict[str, str]):
    inference_response = InferenceResponse(
        model_name="foo", outputs=[], parameters=parameters
    )
    headers = extract_headers(inference_response)

    assert headers == expected
    if inference_response.parameters:
        assert inference_response.parameters.headers is None


def _check_uvloop_availability():
    avail = True
    try:
        import uvloop  # noqa: F401
    except ImportError:  # pragma: no cover
        avail = False
    return avail


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.7.1+rhaiv.8", "1.7.1"),
        ("1.7.1", "1.7.1"),
        ("1.7.0.dev0", "1.7.0.dev0"),
    ],
)
def test_get_normalized_version(version: str | None, expected: str):
    assert get_normalized_version(version) == expected


def test_get_normalized_version_default_uses_current_version():
    assert get_normalized_version() == __version__.split("+", 1)[0]


def test_uvloop_auto_install():
    uvloop_available = _check_uvloop_availability()
    install_uvloop_event_loop()
    policy = asyncio.get_event_loop_policy()

    if uvloop_available:
        assert type(policy).__module__.startswith("uvloop")
    else:
        if platform.system() == "Windows":
            assert isinstance(policy, asyncio.WindowsProactorEventLoopPolicy)
        elif platform.python_implementation() != "CPython":
            assert isinstance(policy, asyncio.DefaultEventLoopPolicy)


async def test_defer_cancellation_waits_for_operation_to_settle():
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()

    async def operation() -> None:
        started.set()
        await finish.wait()
        completed.set()

    task = asyncio.create_task(defer_cancellation(operation)())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed.is_set()


async def test_defer_cancellation_preserves_repeated_cancellation():
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()

    async def operation() -> None:
        started.set()
        await finish.wait()
        completed.set()

    task = asyncio.create_task(defer_cancellation(operation)())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed.is_set()


async def test_defer_cancellation_reports_operation_failure_after_cancellation(
    caplog: pytest.LogCaptureFixture,
):
    started = asyncio.Event()
    fail = asyncio.Event()

    async def operation() -> None:
        started.set()
        await fail.wait()
        raise RuntimeError("operation failed")

    task = asyncio.create_task(defer_cancellation(operation)())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    fail.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert "Model operation failed after caller cancellation" in caplog.text


async def test_defer_cancellation_returns_operation_result():
    async def operation() -> str:
        return "complete"

    result = await defer_cancellation(operation)()

    assert result == "complete"


async def test_defer_cancellation_propagates_failure_without_cancellation():
    error = RuntimeError("operation failed")

    async def operation() -> None:
        raise error

    with pytest.raises(RuntimeError) as raised:
        await defer_cancellation(operation)()

    assert raised.value is error


async def test_with_operation_lock_serializes_same_lock_operations():
    lock = asyncio.Lock()
    active = 0
    maximum_active = 0

    @with_operation_lock(lambda: lock)
    async def operation() -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1

    await asyncio.gather(operation(), operation())

    assert maximum_active == 1


async def test_with_operation_lock_allows_independent_locks_and_resolvers():
    locks = {"sync": asyncio.Lock(), "async": asyncio.Lock()}
    active = 0
    maximum_active = 0

    def resolve_sync_lock(key: str) -> asyncio.Lock:
        return locks[key]

    async def resolve_async_lock(key: str) -> asyncio.Lock:
        return locks[key]

    @with_operation_lock(resolve_sync_lock)
    async def operation_with_sync_resolver(key: str) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1

    @with_operation_lock(resolve_async_lock)
    async def operation_with_async_resolver(key: str) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1

    await asyncio.gather(
        operation_with_sync_resolver("sync"),
        operation_with_async_resolver("async"),
    )

    assert maximum_active == 2


async def test_composed_lock_and_deferred_cancellation_cancel_while_waiting():
    lock = asyncio.Lock()
    lock_resolved = asyncio.Event()
    entered = asyncio.Event()
    await lock.acquire()

    def lock_for() -> asyncio.Lock:
        lock_resolved.set()
        return lock

    @with_operation_lock(lock_for)
    @defer_cancellation
    async def operation() -> None:
        entered.set()

    task = asyncio.create_task(operation())
    await lock_resolved.wait()
    task.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        lock.release()

    assert not entered.is_set()


async def test_composed_lock_and_deferred_cancellation_after_lock_acquisition():
    lock = asyncio.Lock()
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()

    @with_operation_lock(lambda: lock)
    @defer_cancellation
    async def operation() -> None:
        started.set()
        await finish.wait()
        completed.set()

    task = asyncio.create_task(operation())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    finish.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert completed.is_set()
    assert not lock.locked()
