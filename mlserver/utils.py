import os
import uuid
import asyncio
import inspect
import urllib.parse

from asyncio import Task
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar
from functools import wraps

from .logging import logger
from .types import InferenceRequest, InferenceResponse, Parameters
from .settings import ModelSettings
from .errors import InvalidModelURI
from .version import __version__


T = TypeVar("T")
P = ParamSpec("P")


async def get_model_uri(
    settings: ModelSettings, wellknown_filenames: list[str] = []
) -> str:
    if not settings.parameters:
        raise InvalidModelURI(settings.name)

    model_uri = settings.parameters.uri
    if not model_uri:
        raise InvalidModelURI(settings.name)

    model_uri_components = urllib.parse.urlparse(model_uri, scheme="file")
    if model_uri_components.scheme != "file":
        return model_uri

    full_model_path = to_absolute_path(settings, model_uri_components.path)
    if os.path.isfile(full_model_path):
        return full_model_path

    if os.path.isdir(full_model_path):
        # If full_model_path is a folder, search for a well-known model filename
        for fname in wellknown_filenames:
            model_path = os.path.join(full_model_path, fname)
            if os.path.isfile(model_path):
                return model_path

        # If none, return the folder
        return full_model_path

    # Otherwise, the uri is neither a file nor a folder
    raise InvalidModelURI(settings.name, full_model_path)


def to_absolute_path(model_settings: ModelSettings, uri: str) -> str:
    source = model_settings._source
    if source is None:
        # Treat path as either absolute or relative to the working directory of
        # the MLServer instance
        return uri

    parent_folder = os.path.dirname(source)
    unnormalised = os.path.join(parent_folder, uri)
    return os.path.normpath(unnormalised)


def get_wrapped_method(f: Callable) -> Callable:
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__  # type: ignore

    return f


def generate_uuid() -> str:
    return str(uuid.uuid4())


def insert_headers(
    inference_request: InferenceRequest, headers: dict[str, str]
) -> InferenceRequest:
    # Ensure parameters are present
    if inference_request.parameters is None:
        inference_request.parameters = Parameters()

    parameters = inference_request.parameters

    if parameters.headers is not None:
        # TODO: Raise warning that headers will be replaced and shouldn't be used
        logger.warning(
            f"There are {len(parameters.headers)} entries present in the"
            "`headers` field of the request `parameters` object."
            "The `headers` field of the `parameters` object "
            "SHOULDN'T BE USED directly."
            "These entries will be replaced by the actual headers (REST, Kafka) "
            "or metadata (gRPC) of the incoming request."
        )

    parameters.headers = headers
    return inference_request


def extract_headers(inference_response: InferenceResponse) -> dict[str, str] | None:
    if inference_response.parameters is None:
        return None

    parameters = inference_response.parameters
    if parameters.headers is None:
        return None

    headers = parameters.headers
    parameters.headers = None
    return headers


def _check_current_event_loop_policy() -> str:
    policy = (
        "uvloop"
        if type(asyncio.get_event_loop_policy()).__module__.startswith("uvloop")
        else "asyncio"
    )
    return policy


def install_uvloop_event_loop():
    if "uvloop" == _check_current_event_loop_policy():
        return

    try:
        import uvloop

        uvloop.install()
    except ImportError:
        # else keep the standard asyncio loop as a fallback
        pass

    policy = _check_current_event_loop_policy()

    logger.debug(f"Using asyncio event-loop policy: {policy}")


def schedule_with_callback(coro, cb) -> Task:
    task = asyncio.create_task(coro)
    task.add_done_callback(cb)
    return task


def get_normalized_version(version: str | None = None) -> str:
    """
    Return a public version string without local build metadata.

    Example:
    - 1.7.1+rhaiv.8 -> 1.7.1
    - 1.7.1 -> 1.7.1
    - 1.7.0.dev0 -> 1.7.0.dev0
    """
    resolved_version = version or __version__
    return resolved_version.split("+", 1)[0]


async def _defer_cancellation(operation: Awaitable[T]) -> T:
    """Wait for an accepted operation to settle before propagating cancellation.

    The operation runs in a shielded task so caller cancellation does not
    interrupt it. If the caller is cancelled, wait for the task to finish,
    log any operation failure, and then re-raise the caller's cancellation.
    """
    task = asyncio.ensure_future(operation)
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            cancellation = cancelled
        except Exception:
            break
    if cancellation is not None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Model operation failed after caller cancellation",
                    exc_info=(type(error), error, error.__traceback__),
                )
        raise cancellation
    return task.result()


def defer_cancellation(
    method: Callable[P, Awaitable[T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Defer caller cancellation until an accepted operation has settled.

    Apply this inside :func:`with_operation_lock` so cancellation remains
    immediate while waiting for the lock, then is deferred after the operation
    has acquired the lock and begun changing lifecycle state.
    """

    @wraps(method)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        return await _defer_cancellation(method(*args, **kwargs))

    return wrapped


def with_operation_lock(
    lock_for: Callable[..., asyncio.Lock | Awaitable[asyncio.Lock]],
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Acquire the operation lock before running the decorated method.

    ``lock_for`` is called for each invocation and may return a lock directly
    or an awaitable that resolves to one.

    Resolution and acquisition remain
    cancellable. To defer cancellation after acceptance, compose this
    decorator outside :func:`defer_cancellation`.
    """

    def decorate(
        method: Callable[P, Awaitable[T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(method)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            lock = lock_for(*args, **kwargs)
            if inspect.isawaitable(lock):
                lock = await lock
            async with lock:
                return await method(*args, **kwargs)

        return wrapped

    return decorate
