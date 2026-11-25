"""
Command-line interface to manage MLServer models.
"""

import click
import asyncio

from functools import wraps

from ..server import MLServer
from ..logging import logger, configure_logger
from ..utils import install_uvloop_event_loop

from .build import (
    build_image,
    DockerBuildContext,
    write_dockerfile,
)
from .serve import load_settings
from ..settings import ALLOWED_MODEL_IMPLEMENTATIONS, canonicalize_runtime_import_path


def click_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def _validate_runtime_build_args(
    dev: bool,
    allow_runtime_import_paths: tuple[str, ...],
    runtime_source_paths: tuple[str, ...],
) -> None:
    """
    Validate mutual exclusivity and requirements for runtime build arguments.

    Raises:
        click.UsageError: If arguments are invalid or incompatible.
    """
    # Validate mutual exclusivity
    if dev and (allow_runtime_import_paths or runtime_source_paths):
        raise click.UsageError(
            "--dev cannot be combined with --allow-runtime or --runtime-path"
        )

    # Reject built-in runtimes in --allow-runtime
    if allow_runtime_import_paths:
        builtin_runtimes = []
        for runtime in allow_runtime_import_paths:
            canonical = canonicalize_runtime_import_path(runtime)
            if canonical in ALLOWED_MODEL_IMPLEMENTATIONS:
                builtin_runtimes.append(runtime)

        if builtin_runtimes:
            builtin_list = ", ".join(f"'{r}'" for r in builtin_runtimes)
            raise click.UsageError(
                f"Built-in runtime(s) {builtin_list} cannot be specified with "
                "--allow-runtime. Built-in runtimes are always allowed by default. "
                "To use a custom runtime implementation, use a different module name "
                "(e.g., 'custom_sklearn.MyRuntime')."
            )

    # Require both flags together for custom runtimes
    if allow_runtime_import_paths and not runtime_source_paths:
        raise click.UsageError("--allow-runtime requires --runtime-path.")
    if runtime_source_paths and not allow_runtime_import_paths:
        raise click.UsageError("--runtime-path requires --allow-runtime.")


def _prepare_docker_build_context(
    folder: str,
    allow_runtime_import_paths: tuple[str, ...],
    runtime_source_paths: tuple[str, ...],
    dev: bool,
) -> DockerBuildContext:
    """Validate arguments and prepare Docker build context.

    Raises:
        click.UsageError: If arguments are invalid or build context creation fails.
    """
    _validate_runtime_build_args(dev, allow_runtime_import_paths, runtime_source_paths)

    try:
        return DockerBuildContext.from_cli_args(
            folder,
            allow_runtime_import_paths,
            runtime_source_paths,
            dev,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@click.group()
@click.version_option()
def root():
    """
    Command-line interface to manage MLServer models.
    """
    pass


@root.command("start")
@click.argument("folder", nargs=1)
@click_async
async def start(folder: str):
    """
    Start serving a machine learning model with MLServer.
    """
    settings, models_settings = await load_settings(folder)

    server = MLServer(settings)
    await server.start(models_settings)


@root.command("build")
@click.argument("folder", nargs=1)
@click.option("-t", "--tag", type=str)
@click.option("--no-cache", default=False, is_flag=True)
@click.option(
    "--allow-runtime",
    "allow_runtime_import_paths",
    multiple=True,
    type=str,
    help=(
        "Additional custom runtime import path to allow in the built image. "
        "Use exact dotted Python import paths (module.ClassName)."
    ),
)
@click.option(
    "--runtime-path",
    "runtime_source_paths",
    multiple=True,
    type=click.Path(path_type=str),
    help=(
        "Path (relative to build folder) to custom runtime Python module/package "
        "to bake into image import path."
    ),
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help=(
        "Build a development image that allows any runtime to be loaded. "
        "Cannot be combined with --allow-runtime or --runtime-path."
    ),
)
@click_async
async def build(
    folder: str,
    tag: str,
    no_cache: bool = False,
    allow_runtime_import_paths: tuple[str, ...] = (),
    runtime_source_paths: tuple[str, ...] = (),
    dev: bool = False,
):
    """
    Build a Docker image for a custom MLServer runtime.
    """
    docker_build_context = _prepare_docker_build_context(
        folder, allow_runtime_import_paths, runtime_source_paths, dev
    )
    build_image(
        docker_build_context.folder,
        docker_build_context.dockerfile,
        tag,
        no_cache=no_cache,
    )
    logger.info(f"Successfully built custom Docker image with tag {tag}")


@root.command("dockerfile")
@click.argument("folder", nargs=1)
@click.option("-i", "--include-dockerignore", is_flag=True)
@click.option(
    "--allow-runtime",
    "allow_runtime_import_paths",
    multiple=True,
    type=str,
    help=(
        "Additional custom runtime import path to include in the generated "
        "Dockerfile allowlist. Use exact dotted Python import paths "
        "(module.ClassName)."
    ),
)
@click.option(
    "--runtime-path",
    "runtime_source_paths",
    multiple=True,
    type=click.Path(path_type=str),
    help=(
        "Path (relative to folder) to custom runtime Python module/package "
        "to include in generated Dockerfile import path."
    ),
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help=(
        "Generate a Dockerfile for a development image that allows any runtime. "
        "Cannot be combined with --allow-runtime or --runtime-path."
    ),
)
@click_async
async def dockerfile(
    folder: str,
    include_dockerignore: bool,
    allow_runtime_import_paths: tuple[str, ...] = (),
    runtime_source_paths: tuple[str, ...] = (),
    dev: bool = False,
):
    """
    Generate a Dockerfile
    """
    docker_build_context = _prepare_docker_build_context(
        folder, allow_runtime_import_paths, runtime_source_paths, dev
    )
    dockerfile_path = write_dockerfile(
        folder,
        docker_build_context.dockerfile,
        include_dockerignore=include_dockerignore,
    )
    logger.info(f"Successfully written Dockerfile in {dockerfile_path}")


def main():
    configure_logger()
    install_uvloop_event_loop()
    root()


if __name__ == "__main__":
    main()
