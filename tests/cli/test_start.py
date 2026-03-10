import os
import pytest
import signal
import sys

from aiohttp.client_exceptions import ClientConnectionError, ClientResponseError
from subprocess import Popen
from typing import Tuple

from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest

from ..utils import RESTClient
from .test_start_cases import case_custom_module, case_sum_model


def _spawn_mlserver(folder: str) -> Popen:
    # Use the same interpreter as the running test env so imports resolve
    # consistently across tox and local runs.
    return Popen(
        [sys.executable, "-m", "mlserver.cli.main", "start", folder],
        start_new_session=True,
    )


def _stop_mlserver(process: Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process may have already exited before fixture teardown runs.
        pass


@pytest.fixture
def settings(settings: Settings, free_ports: Tuple[int, int]) -> Settings:
    http_port, grpc_port, metrics_port = free_ports

    settings.http_port = http_port
    settings.grpc_port = grpc_port
    settings.metrics_port = metrics_port

    return settings


@pytest.fixture
def mlserver_start_sum_model(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Popen:
    # Baseline scenario: importable runtime (`tests.fixtures.SumModel`).
    sum_model_folder = case_sum_model(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(sum_model_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
def mlserver_start_custom_module(
    tmp_path: str, settings: Settings, sum_model_settings: ModelSettings
) -> Popen:
    # Security scenario: model-folder module (`custom.SumModel`) should not load.
    custom_module_folder = case_custom_module(tmp_path, settings, sum_model_settings)
    p = _spawn_mlserver(custom_module_folder)

    yield p

    _stop_mlserver(p)


@pytest.fixture
async def rest_client(settings: Settings) -> RESTClient:
    http_server = f"127.0.0.1:{settings.http_port}"
    client = RESTClient(http_server)

    yield client

    await client.close()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_live(rest_client: RESTClient):
    await rest_client.wait_until_live()
    is_live = await rest_client.live()
    assert is_live

    # Assert that the server is live, but some models are still loading
    with pytest.raises(ClientResponseError):
        await rest_client.ready()


@pytest.mark.usefixtures("mlserver_start_sum_model")
async def test_infer(
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    await rest_client.wait_until_model_ready(sum_model_settings.name)
    response = await rest_client.infer(sum_model_settings.name, inference_request)

    assert len(response.outputs) == 1


@pytest.mark.usefixtures("mlserver_start_custom_module")
async def test_custom_module_fails_closed(
    rest_client: RESTClient,
    sum_model_settings: ModelSettings,
):
    # Fail closed when runtime points to a non-importable model-folder module.
    with pytest.raises((ClientResponseError, ClientConnectionError)):
        await rest_client.wait_until_model_ready(sum_model_settings.name)
