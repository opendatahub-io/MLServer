"""
Parallel inference tests configuration.

IMPORTANT: These tests spawn worker processes using multiprocessing.Process.
Some tests require DEVELOPMENT mode to use custom environments (environment_tarball).

## How Workers Inherit Security Mode:

Production:
  - Workers read /etc/mlserver/trusted-runtimes.json (fixed path)
  - Main process and workers naturally agree on PRODUCTION mode

Tests:
  - Test artifact is in random temp path (e.g., /tmp/trusted-runtimes-xyz.json)
  - Main process uses monkeypatch to override path (in-memory, not inherited)
  - Workers spawn as new processes, don't inherit monkeypatch
  - Solution: Set TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV environment variable
  - Workers' sitecustomize.py (see conftest.py) reads env var and overrides path

## Test Isolation:

These tests MUST run sequentially (not with pytest -n auto) because:
  - Fixtures modify shared environment variable (TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV)
  - Running in parallel would cause race conditions between tests
  - tox.ini runs parallel tests separately without -n auto (safe)

If you run manually: pytest tests/parallel/ (sequential, safe)
DO NOT RUN: pytest tests/parallel/ -n auto (parallel, UNSAFE - race conditions)
"""

import asyncio
import os
import pytest

from multiprocessing import Queue

from mlserver.settings import Settings, ModelSettings, ModelParameters
from mlserver.types import InferenceRequest
from mlserver.utils import generate_uuid
from mlserver.model import MLModel
from mlserver.env import Environment
from mlserver.parallel.dispatcher import Dispatcher
from mlserver.parallel.model import ModelMethods
from mlserver.parallel.pool import InferencePool, _spawn_worker
from mlserver.parallel.worker import Worker
from mlserver.parallel.utils import configure_inference_pool, cancel_task
from mlserver.parallel.messages import (
    ModelUpdateMessage,
    ModelUpdateType,
    ModelRequestMessage,
)

from ..fixtures import ErrorModel, EnvModel


@pytest.fixture(autouse=True)
def sync_development_mode_to_workers(request, tmp_path):
    """Auto-sync development_mode fixture to spawned workers via env var.

    When a test uses the development_mode fixture, this automatically propagates
    DEVELOPMENT mode to spawned worker processes by setting the environment variable
    that workers' sitecustomize.py reads.

    This fixture runs automatically for all tests in tests/parallel/ but only modifies
    the environment when development_mode is actually being used.

    Why this is needed:
    - Main process: development_mode fixture uses monkeypatch (in-memory only)
    - Spawned workers: New processes that need real environment variables
    - This bridges the gap by setting the env var workers need
    """
    from conftest import TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV

    # Only set env var if test is using development_mode fixture
    if "development_mode" not in request.fixturenames:
        yield
        return

    # Point to non-existent file = DEVELOPMENT mode (no trusted-runtimes.json)
    # Workers' sitecustomize.py reads this env var to override artifact path
    non_existent = str(tmp_path / "does-not-exist.json")
    original_env = os.environ.get(TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV)
    os.environ[TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV] = non_existent

    try:
        yield
    finally:
        # Restore original env var
        if original_env is not None:
            os.environ[TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV] = original_env
        else:
            os.environ.pop(TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV, None)


@pytest.fixture
async def inference_pool(settings: Settings) -> InferencePool:
    pool = InferencePool(settings)
    yield pool

    await pool.close()


@pytest.fixture
async def dispatcher(inference_pool) -> Dispatcher:
    return inference_pool._dispatcher


@pytest.fixture
async def error_model(inference_pool: InferencePool, error_model: MLModel) -> MLModel:
    model = await inference_pool.load_model(error_model)

    yield model

    await inference_pool.unload_model(error_model)


@pytest.fixture
async def load_error_model() -> MLModel:
    error_model_settings = ModelSettings(
        name="foo",
        implementation=ErrorModel,
        parameters=ModelParameters(load_error=True),
    )
    error_model = ErrorModel(error_model_settings)

    yield error_model


@pytest.fixture
def settings(settings: Settings, tmp_path: str) -> Settings:
    settings.parallel_workers = 2
    settings.environments_dir = str(tmp_path)

    configure_inference_pool(settings)
    return settings


@pytest.fixture
async def responses(settings: Settings) -> Queue:
    # NOTE: This fixture depends on settings to ensure the multiprocessing
    # context has been set ahead of time to `spawn` (handled in the
    # configure_inference_pool method)
    q = Queue()
    yield q

    q.close()


@pytest.fixture
async def worker(
    settings: Settings,
    event_loop: asyncio.AbstractEventLoop,
    responses: Queue,
    load_message: ModelUpdateMessage,
) -> Worker:
    worker = Worker(settings, responses)

    # Simulate the worker running on a different process, but keep it to a
    # thread to simplify debugging.
    # Note that we call `worker.coro_run` instead of `worker.run` to avoid also
    # triggering the other set up methods of `worker.run`.
    worker_task = event_loop.run_in_executor(
        None, lambda: asyncio.run(worker.coro_run())
    )

    # Send an update and wait for its response (although we ignore it)
    worker.send_update(load_message)
    responses.get()

    yield worker

    await worker.stop()
    await cancel_task(worker_task)


@pytest.fixture
def load_message(sum_model_settings: ModelSettings) -> ModelUpdateMessage:
    return ModelUpdateMessage(
        update_type=ModelUpdateType.Load, model_settings=sum_model_settings
    )


@pytest.fixture
def unload_message(sum_model_settings: ModelSettings) -> ModelUpdateMessage:
    return ModelUpdateMessage(
        update_type=ModelUpdateType.Unload, model_settings=sum_model_settings
    )


@pytest.fixture
def inference_request_message(
    sum_model_settings: ModelSettings, inference_request: InferenceRequest
) -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=sum_model_settings.name,
        model_version=sum_model_settings.parameters.version,
        method_name=ModelMethods.Predict.value,
        method_args=[inference_request],
    )


@pytest.fixture
def metadata_request_message(sum_model_settings: ModelSettings) -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=sum_model_settings.name,
        model_version=sum_model_settings.parameters.version,
        method_name=ModelMethods.Metadata.value,
    )


@pytest.fixture
def custom_request_message(sum_model_settings: ModelSettings) -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=sum_model_settings.name,
        model_version=sum_model_settings.parameters.version,
        # From `SumModel` class in tests/fixtures.py
        method_name="my_payload",
        method_kwargs={"payload": [1, 2, 3]},
    )


@pytest.fixture
def env_model_settings(development_mode, env_tarball: str) -> ModelSettings:
    """Model settings with environment_tarball (requires DEVELOPMENT mode).

    The development_mode fixture enables DEVELOPMENT mode for the main process.
    The sync_development_mode_to_workers autouse fixture automatically propagates
    this to spawned workers via environment variable.

    Args:
        development_mode: Fixture dependency (ensures DEVELOPMENT mode is active)
        env_tarball: Path to environment tarball
    """
    return ModelSettings(
        name="env-model",
        implementation=EnvModel,
        parameters=ModelParameters(environment_tarball=env_tarball),
    )


@pytest.fixture
def existing_env_model_settings(
    development_mode, env_tarball: str, tmp_path
) -> ModelSettings:
    """Model settings with environment_path (requires DEVELOPMENT mode).

    The development_mode fixture enables DEVELOPMENT mode for the main process.
    The sync_development_mode_to_workers autouse fixture automatically propagates
    this to spawned workers via environment variable.

    Args:
        development_mode: Fixture dependency (ensures DEVELOPMENT mode is active)
        env_tarball: Path to environment tarball
        tmp_path: pytest tmp_path for extraction
    """
    from mlserver.env import _extract_env

    env_path = str(tmp_path)
    _extract_env(env_tarball, env_path)

    return ModelSettings(
        name="exising_env_model",
        implementation=EnvModel,
        parameters=ModelParameters(environment_path=env_path),
    )


@pytest.fixture
async def worker_with_env(
    settings: Settings,
    responses: Queue,
    env: Environment,
    env_model_settings: ModelSettings,
):
    # NOTE: This fixture will start an actual worker running on a separate
    # process.
    worker = _spawn_worker(settings, responses, env)

    load_message = ModelUpdateMessage(
        update_type=ModelUpdateType.Load, model_settings=env_model_settings
    )
    worker.send_update(load_message)
    responses.get()

    yield worker

    await worker.stop()
