import os
import uuid
import asyncio
import urllib.parse

from asyncio import Task
from collections.abc import Callable

from .logging import logger
from .types import InferenceRequest, InferenceResponse, Parameters
from .settings import ModelSettings
from .errors import InvalidModelURI
from .version import __version__


async def get_model_uri(
    settings: ModelSettings, wellknown_filenames: list[str] = []
) -> str:
    """Resolve the model artifact URI from settings to an absolute file path.

    For ``file://`` URIs (the default), resolves the path relative to the
    settings source, checks for well-known filenames inside directories,
    and raises :class:`InvalidModelURI` if nothing is found.
    Non-file schemes are returned as-is.
    """
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
    """Convert a relative model URI to an absolute path based on the settings source.

    If the settings have no ``_source`` (e.g. created programmatically), the
    URI is returned unchanged and treated as relative to the working directory.
    """
    source = model_settings._source
    if source is None:
        # Treat path as either absolute or relative to the working directory of
        # the MLServer instance
        return uri

    parent_folder = os.path.dirname(source)
    unnormalised = os.path.join(parent_folder, uri)
    return os.path.normpath(unnormalised)


def get_wrapped_method(f: Callable) -> Callable:
    """Unwrap a decorated callable by following the ``__wrapped__`` chain."""
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__  # type: ignore

    return f


def generate_uuid() -> str:
    """Generate a random UUID4 string for use as a unique identifier."""
    return str(uuid.uuid4())


def insert_headers(
    inference_request: InferenceRequest, headers: dict[str, str]
) -> InferenceRequest:
    """Attach transport-level headers (REST, gRPC, Kafka) into the request's
    ``parameters.headers`` field, replacing any existing entries."""
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
    """Remove and return the headers dict from a response's parameters.

    Returns ``None`` if no headers are present. The headers field on the
    response is cleared after extraction.
    """
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
    """Install uvloop as the asyncio event-loop policy if available.

    Falls back silently to the default asyncio loop when uvloop is not
    installed.
    """
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
    """Create an asyncio task from a coroutine and attach a done-callback."""
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
