import subprocess
import os
import json

from tempfile import TemporaryDirectory
from typing import List, Optional

from .. import __version__
from ..logging import logger
from ..settings import TRUSTED_RUNTIMES_ARTIFACT_PATH, is_valid_runtime_import_path

from .constants import (
    DockerfileName,
    DockerfileTemplate,
    DockerignoreName,
    Dockerignore,
    DefaultBaseImage,
)


def get_invalid_runtime_import_paths(
    custom_runtimes: Optional[List[str]],
) -> List[str]:
    if not custom_runtimes:
        return []

    invalid_runtimes: List[str] = []
    for runtime in custom_runtimes:
        runtime = runtime.strip()
        if not is_valid_runtime_import_path(runtime):
            invalid_runtimes.append(runtime)

    return sorted(set(invalid_runtimes))


def _normalise_custom_runtimes(custom_runtimes: Optional[List[str]]) -> List[str]:
    if not custom_runtimes:
        return []

    invalid_runtimes = get_invalid_runtime_import_paths(custom_runtimes)
    if invalid_runtimes:
        invalid_values = ", ".join(invalid_runtimes)
        raise ValueError(
            f"Invalid runtime import path(s): {invalid_values}. "
            "Expected a dotted Python import path."
        )

    normalised: List[str] = []
    seen = set()
    for runtime in custom_runtimes:
        runtime = runtime.strip()
        if runtime not in seen:
            seen.add(runtime)
            normalised.append(runtime)

    return normalised


def generate_dockerfile(
    base_image: str = DefaultBaseImage,
    custom_runtimes: Optional[List[str]] = None,
) -> str:
    base_image = base_image.format(version=__version__)
    trusted_runtime_allowlist_json = json.dumps(
        _normalise_custom_runtimes(custom_runtimes),
        indent=2,
    )
    return DockerfileTemplate.format(
        base_image=base_image,
        trusted_runtime_allowlist_json=trusted_runtime_allowlist_json,
        trusted_runtimes_artifact_path=TRUSTED_RUNTIMES_ARTIFACT_PATH,
    )


def write_dockerfile(
    folder: str, dockerfile: str, include_dockerignore: bool = True
) -> str:
    dockerfile_path = os.path.join(folder, DockerfileName)
    with open(dockerfile_path, "w") as dockerfile_handler:
        logger.info(f"Writing Dockerfile in {dockerfile_path}")
        dockerfile_handler.write(dockerfile)

    if include_dockerignore:
        # Point to our own .dockerignore
        # https://docs.docker.com/engine/reference/commandline/build/#use-a-dockerignore-file
        dockerignore_path = dockerfile_path + DockerignoreName
        with open(dockerignore_path, "w") as dockerignore_handler:
            logger.info(f"Writing .dockerignore in {dockerignore_path}")
            dockerignore_handler.write(Dockerignore)

    return dockerfile_path


def build_image(
    folder: str, dockerfile: str, image_tag: str, no_cache: bool = False
) -> str:
    logger.info(f"Building Docker image with tag {image_tag}")
    _docker_command_prefix = "docker build --rm "
    with TemporaryDirectory() as tmp_dir:
        dockerfile_path = write_dockerfile(tmp_dir, dockerfile)
        _docker_command_suffix = f"{folder} -f {dockerfile_path} -t {image_tag}"
        if no_cache:
            build_cmd = _docker_command_prefix + "--no-cache " + _docker_command_suffix
        else:
            build_cmd = _docker_command_prefix + _docker_command_suffix
        build_env = os.environ.copy()
        build_env["DOCKER_BUILDKIT"] = "1"
        subprocess.run(build_cmd, check=True, shell=True, env=build_env)

    return image_tag
