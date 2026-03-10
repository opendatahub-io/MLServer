import logging

import pytest
import random

from docker.client import DockerClient
from pytest_cases import fixture, parametrize_with_cases
from pathlib import Path
from typing import Tuple, Optional

from mlserver import __version__
from mlserver.repository import DEFAULT_MODEL_SETTINGS_FILENAME
from mlserver.repository.load import load_model_settings
from mlserver.types import InferenceRequest, Parameters
from mlserver.settings import Settings, TRUSTED_RUNTIMES_ARTIFACT_PATH
from mlserver.cli.constants import DockerfileTemplate, DefaultBaseImage
from mlserver.cli.build import generate_dockerfile, build_image

from ..utils import RESTClient


@fixture
@parametrize_with_cases("custom_runtime_path")
def custom_image(
    docker_client: DockerClient, custom_runtime_path: str, current_cases
) -> str:
    discovered_implementations = []
    settings_paths = sorted(
        Path(custom_runtime_path).rglob(DEFAULT_MODEL_SETTINGS_FILENAME)
    )
    for settings_path in settings_paths:
        model_settings = load_model_settings(str(settings_path))
        discovered_implementations.append(model_settings.implementation_)
    custom_runtime_implementations = sorted(set(discovered_implementations))

    dockerfile = generate_dockerfile(custom_runtimes=custom_runtime_implementations)
    current_case = current_cases["custom_image"]["custom_runtime_path"]
    image_name = f"{current_case.id}:0.1.0"
    build_image(custom_runtime_path, dockerfile, image_name)

    yield image_name

    # in CI sometimes this fails, TODO: indentify why
    try:
        docker_client.images.remove(image=image_name, force=True)
    except Exception:
        logging.warning("skipping remove")


@pytest.fixture
def random_user_id() -> int:
    return random.randint(1000, 65536)


@pytest.fixture
def custom_runtime_server(
    docker_client: DockerClient,
    custom_image: str,
    settings: Settings,
    free_ports: Tuple[int, int, int],
    random_user_id: int,
) -> str:
    host_http_port, host_grpc_port, host_metrics_port = free_ports

    container = docker_client.containers.run(
        custom_image,
        ports={
            f"{settings.http_port}/tcp": str(host_http_port),
            f"{settings.grpc_port}/tcp": str(host_grpc_port),
            f"{settings.metrics_port}/tcp": str(host_metrics_port),
        },
        detach=True,
        user=random_user_id,
        environment={"MLSERVER_MODELS_DIR": "."},
    )

    yield f"127.0.0.1:{host_http_port}", f"127.0.0.1:{host_grpc_port}"

    container.remove(force=True)


@pytest.fixture
def custom_runtime_model_settings(custom_image_custom_runtime_path: str):
    settings_paths = sorted(
        Path(custom_image_custom_runtime_path).rglob(DEFAULT_MODEL_SETTINGS_FILENAME)
    )
    if not settings_paths:
        raise FileNotFoundError(
            f"Could not find {DEFAULT_MODEL_SETTINGS_FILENAME} under "
            f"{custom_image_custom_runtime_path}"
        )

    return load_model_settings(str(settings_paths[0]))


@pytest.mark.parametrize(
    "base_image",
    [
        None,
        "customreg/customimage:{version}-slim",
        "customreg/custonimage:customtag",
    ],
)
def test_generate_dockerfile(base_image: Optional[str]):
    dockerfile = ""
    if base_image is None:
        dockerfile = generate_dockerfile()
        base_image = DefaultBaseImage
    else:
        dockerfile = generate_dockerfile(base_image=base_image)

    expected = base_image.format(version=__version__)
    assert expected in dockerfile
    assert dockerfile == DockerfileTemplate.format(
        base_image=expected,
        trusted_runtime_allowlist_json="[]",
        trusted_runtimes_artifact_path=TRUSTED_RUNTIMES_ARTIFACT_PATH,
    )


def test_generate_dockerfile_with_custom_runtime_allowlist():
    dockerfile = generate_dockerfile(
        custom_runtimes=["custom.MyRuntime", "custom.MyRuntime", "other.Runtime"]
    )

    assert TRUSTED_RUNTIMES_ARTIFACT_PATH in dockerfile
    assert '"custom.MyRuntime"' in dockerfile
    assert '"other.Runtime"' in dockerfile
    assert dockerfile.count('"custom.MyRuntime"') == 1


def test_generate_dockerfile_rejects_invalid_runtime_path():
    with pytest.raises(ValueError, match="Invalid runtime import path"):
        generate_dockerfile(custom_runtimes=["invalid-runtime"])


def test_build(docker_client: DockerClient, custom_image: str):
    image = docker_client.images.get(custom_image)
    assert image.tags == [custom_image]


async def test_infer_custom_runtime(
    custom_runtime_server: Tuple[str, str],
    custom_runtime_model_settings,
    inference_request: InferenceRequest,
):
    http_server, _ = custom_runtime_server
    rest_client = RESTClient(http_server)
    try:
        model_name = custom_runtime_model_settings.name

        await rest_client.wait_until_ready()

        await rest_client.wait_until_model_indexed(model_name)

        inference_request.inputs[0].parameters = Parameters(content_type="np")
        inference_response = await rest_client.infer(model_name, inference_request)
        assert len(inference_response.outputs) == 1
    finally:
        await rest_client.close()
