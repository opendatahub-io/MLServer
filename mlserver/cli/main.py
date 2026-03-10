"""
Command-line interface to manage MLServer models.
"""

import click
import asyncio
import json

from functools import wraps
from pathlib import Path

from .init_project import init_cookiecutter_project

from ..server import MLServer
from ..logging import logger, configure_logger
from ..utils import install_uvloop_event_loop
from ..repository import DEFAULT_MODEL_SETTINGS_FILENAME
from ..settings import ALLOWED_MODEL_IMPLEMENTATIONS, is_valid_runtime_import_path

from .build import (
    generate_dockerfile,
    build_image,
    write_dockerfile,
    get_invalid_runtime_import_paths,
)
from .serve import load_settings
from ..batch_processing import process_batch, CHOICES_TRANSPORT


def click_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def _validate_custom_runtime_allowlist(folder: str, allow_runtimes=()) -> None:
    declared_custom_runtimes = set(allow_runtimes)
    effective_allowlist = ALLOWED_MODEL_IMPLEMENTATIONS.union(declared_custom_runtimes)
    missing_runtime_declarations = []
    invalid_declared_runtimes = get_invalid_runtime_import_paths(list(allow_runtimes))
    if invalid_declared_runtimes:
        invalid_values = ", ".join(invalid_declared_runtimes)
        raise click.UsageError(
            "Invalid --allow-runtime value(s): "
            f"{invalid_values}. Expected format: module.ClassName"
        )

    model_settings_paths = Path(folder).rglob(DEFAULT_MODEL_SETTINGS_FILENAME)
    for model_settings_path in model_settings_paths:
        with open(model_settings_path, "r", encoding="utf-8") as f:
            try:
                model_settings = json.load(f)
            except json.JSONDecodeError as exc:
                raise click.UsageError(
                    "Invalid JSON in "
                    f"{model_settings_path}:{exc.lineno}:{exc.colno} - {exc.msg}"
                ) from exc
        if not isinstance(model_settings, dict):
            raise click.UsageError(
                f"Invalid JSON schema in {model_settings_path}: "
                "expected a JSON object."
            )

        implementation = model_settings.get("implementation")
        if implementation is None:
            continue
        if not is_valid_runtime_import_path(implementation):
            raise click.UsageError(
                f"Invalid implementation in {model_settings_path}: "
                f"{implementation!r}. Expected format: module.ClassName"
            )

        if implementation not in effective_allowlist:
            missing_runtime_declarations.append((model_settings_path, implementation))

    if missing_runtime_declarations:
        missing_lines = [
            f"- {path}: {implementation}"
            for path, implementation in missing_runtime_declarations
        ]
        missing_details = "\n".join(missing_lines)
        missing_implementations = sorted(
            {implementation for _, implementation in missing_runtime_declarations}
        )
        suggested_flags = " ".join(
            f"--allow-runtime {implementation}"
            for implementation in missing_implementations
        )
        raise click.UsageError(
            "Found non-built-in model implementations not declared with "
            "`--allow-runtime`:\n"
            f"{missing_details}\n\n"
            "Suggested flags:\n"
            f"{suggested_flags}"
        )


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
    "allow_runtimes",
    multiple=True,
    type=str,
    help=(
        "Additional custom runtime import path to allow in the built image. "
        "Use exact dotted Python import paths (module.ClassName)."
    ),
)
@click_async
async def build(
    folder: str,
    tag: str,
    no_cache: bool = False,
    allow_runtimes=(),
):
    """
    Build a Docker image for a custom MLServer runtime.
    """
    _validate_custom_runtime_allowlist(folder, allow_runtimes)
    dockerfile = generate_dockerfile(custom_runtimes=list(allow_runtimes))
    build_image(folder, dockerfile, tag, no_cache=no_cache)
    logger.info(f"Successfully built custom Docker image with tag {tag}")


@root.command("init")
# TODO: Update to have template(s) in the SeldonIO org
@click.option("-t", "--template", default="https://github.com/EthicalML/sml-security/")
@click_async
async def init_project(template: str):
    """
    Generate a base project template
    """
    init_cookiecutter_project(template)


@root.command("dockerfile")
@click.argument("folder", nargs=1)
@click.option("-i", "--include-dockerignore", is_flag=True)
@click.option(
    "--allow-runtime",
    "allow_runtimes",
    multiple=True,
    type=str,
    help=(
        "Additional custom runtime import path to include in the generated "
        "Dockerfile allowlist. Use exact dotted Python import paths "
        "(module.ClassName)."
    ),
)
@click_async
async def dockerfile(folder: str, include_dockerignore: bool, allow_runtimes=()):
    """
    Generate a Dockerfile
    """
    _validate_custom_runtime_allowlist(folder, allow_runtimes)
    dockerfile = generate_dockerfile(custom_runtimes=list(allow_runtimes))
    dockerfile_path = write_dockerfile(
        folder, dockerfile, include_dockerignore=include_dockerignore
    )
    logger.info(f"Successfully written Dockerfile in {dockerfile_path}")


@root.command("infer")
@click.option(
    "--url",
    "-u",
    default="localhost:8080",
    envvar="MLSERVER_INFER_URL",
    help=(
        "URL of the MLServer to send inference requests to. "
        "Should not contain http or https."
    ),
)
@click.option(
    "--model-name",
    "-m",
    type=str,
    required=True,
    envvar="MLSERVER_INFER_MODEL_NAME",
    help="Name of the model to send inference requests to.",
)
@click.option(
    "--input-data-path",
    "-i",
    required=True,
    type=click.Path(),
    envvar="MLSERVER_INFER_INPUT_DATA_PATH",
    help="Local path to the input file containing inference requests to be processed.",
)
@click.option(
    "--output-data-path",
    "-o",
    required=True,
    type=click.Path(),
    envvar="MLSERVER_INFER_OUTPUT_DATA_PATH",
    help="Local path to the output file for the inference responses to be  written to.",
)
@click.option("--workers", "-w", default=10, envvar="MLSERVER_INFER_WORKERS")
@click.option("--retries", "-r", default=3, envvar="MLSERVER_INFER_RETRIES")
@click.option(
    "--batch-size",
    "-s",
    default=1,
    envvar="MLSERVER_INFER_BATCH_SIZE",
    help="Send inference requests grouped together as micro-batches.",
)
@click.option(
    "--binary-data",
    "-b",
    is_flag=True,
    default=False,
    envvar="MLSERVER_INFER_BINARY_DATA",
    help="Send inference requests as binary data (not fully supported).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    envvar="MLSERVER_INFER_VERBOSE",
    help="Verbose mode.",
)
@click.option(
    "--extra-verbose",
    "-vv",
    is_flag=True,
    default=False,
    envvar="MLSERVER_INFER_EXTRA_VERBOSE",
    help="Extra verbose mode (shows detailed requests and responses).",
)
@click.option(
    "--transport",
    "-t",
    envvar="MLSERVER_INFER_TRANSPORT",
    type=click.Choice(CHOICES_TRANSPORT),
    default="rest",
    help=(
        "Transport type to use to send inference requests. "
        "Can be 'rest' or 'grpc' (not yet supported)."
    ),
)
@click.option(
    "--request-headers",
    "-H",
    envvar="MLSERVER_INFER_REQUEST_HEADERS",
    type=str,
    multiple=True,
    help=(
        "Headers to be set on each inference request send to the server. "
        "Multiple options are allowed as: -H 'Header1: Val1' -H 'Header2: Val2'. "
        "When setting up as environmental provide as 'Header1:Val1 Header2:Val2'."
    ),
)
@click.option(
    "--timeout",
    default=60,
    envvar="MLSERVER_INFER_CONNECTION_TIMEOUT",
    help="Connection timeout to be passed to tritonclient.",
)
@click.option(
    "--batch-interval",
    default=0,
    type=float,
    envvar="MLSERVER_INFER_BATCH_INTERVAL",
    help="Minimum time interval (in seconds) between requests made by each worker.",
)
@click.option(
    "--batch-jitter",
    default=0,
    type=float,
    envvar="MLSERVER_INFER_BATCH_JITTER",
    help="Maximum random jitter (in seconds) added to batch interval between requests.",
)
@click.option(
    "--use-ssl",
    is_flag=True,
    default=False,
    envvar="MLSERVER_INFER_USE_SSL",
    help="Use SSL in communications with inference server.",
)
@click.option(
    "--insecure",
    is_flag=True,
    default=False,
    envvar="MLSERVER_INFER_INSECURE",
    help="Disable SSL verification in communications. Use with caution.",
)
@click_async
# TODO: add flags for SSL --key-file and --cert-file (see Tritonclient examples)
async def infer(
    model_name,
    url,
    workers,
    retries,
    batch_size,
    input_data_path,
    output_data_path,
    binary_data,
    transport,
    request_headers,
    timeout,
    batch_interval,
    batch_jitter,
    use_ssl,
    insecure,
    verbose,
    extra_verbose,
):
    """
    Deprecated: This experimental feature will be removed in future work.
    Execute batch inference requests against V2 inference server.
    """
    logger.warn(
        "This feature has been deprecated and will be removed in a future version"
    )

    await process_batch(
        model_name=model_name,
        url=url,
        workers=workers,
        retries=retries,
        batch_size=batch_size,
        input_data_path=input_data_path,
        output_data_path=output_data_path,
        binary_data=binary_data,
        transport=transport,
        request_headers=request_headers,
        timeout=timeout,
        batch_interval=batch_interval,
        batch_jitter=batch_jitter,
        use_ssl=use_ssl,
        insecure=insecure,
        verbose=verbose,
        extra_verbose=extra_verbose,
    )


def main():
    configure_logger()
    install_uvloop_event_loop()
    root()


if __name__ == "__main__":
    main()
