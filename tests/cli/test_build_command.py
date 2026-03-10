import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

DEFAULT_IMAGE_TAG = "my-custom-image:0.1.0"


@pytest.fixture
def cli_main():
    return importlib.import_module("mlserver.cli.main")


@pytest.fixture
def runner():
    return CliRunner()


def _patch_build_pipeline(monkeypatch, cli_main, captured=None):
    def _fake_generate_dockerfile(*args, **kwargs):
        if captured is not None:
            captured["custom_runtimes"] = kwargs.get("custom_runtimes")
        return "FROM test"

    def _fake_build_image(folder, dockerfile, image_tag, no_cache):
        if captured is not None:
            captured["folder"] = folder
            captured["dockerfile"] = dockerfile
            captured["image_tag"] = image_tag
            captured["no_cache"] = no_cache
        return image_tag

    monkeypatch.setattr(cli_main, "generate_dockerfile", _fake_generate_dockerfile)
    monkeypatch.setattr(cli_main, "build_image", _fake_build_image)


def _write_model_settings_batch(models):
    for rel_path, payload in models.items():
        path = Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def _invoke_build(runner: CliRunner, cli_main, allow_runtimes=()):
    args = ["build", ".", "-t", DEFAULT_IMAGE_TAG]
    for runtime in allow_runtimes:
        args.extend(["--allow-runtime", runtime])

    return runner.invoke(cli_main.root, args)


def _invoke_dockerfile(runner: CliRunner, cli_main, allow_runtimes=()):
    args = ["dockerfile", "."]
    for runtime in allow_runtimes:
        args.extend(["--allow-runtime", runtime])

    return runner.invoke(cli_main.root, args)


def test_build_passes_allow_runtime_to_dockerfile_generator(
    monkeypatch, cli_main, runner
):
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, captured)

    with runner.isolated_filesystem():
        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime", "another.Runtime"),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == ["custom.Runtime", "another.Runtime"]
    assert captured["image_tag"] == DEFAULT_IMAGE_TAG


def test_build_fails_when_custom_implementation_is_not_allowlisted(
    monkeypatch, cli_main, runner
):
    _patch_build_pipeline(monkeypatch, cli_main)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/custom/model-settings.json": {
                    "name": "custom-model",
                    "implementation": "custom.MyRuntime",
                }
            }
        )

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Found non-built-in model implementations" in result.output
    assert "custom.MyRuntime" in result.output
    assert "--allow-runtime custom.MyRuntime" in result.output


def test_build_allows_multiple_model_settings_when_all_custom_are_allowlisted(
    monkeypatch, cli_main, runner
):
    captured = {}
    _patch_build_pipeline(monkeypatch, cli_main, captured)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/a/model-settings.json": {
                    "name": "model-a",
                    "implementation": "custom.RuntimeA",
                },
                "models/b/v2/model-settings.json": {
                    "name": "model-b",
                    "implementation": "custom.RuntimeB",
                },
            }
        )

        result = _invoke_build(
            runner,
            cli_main,
            allow_runtimes=("custom.RuntimeA", "custom.RuntimeB"),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == ["custom.RuntimeA", "custom.RuntimeB"]


@pytest.mark.parametrize("invalid_runtime", ["invalid-runtime", "_private.Runtime"])
def test_build_fails_when_allow_runtime_format_is_invalid(
    monkeypatch, cli_main, runner, invalid_runtime
):
    _patch_build_pipeline(monkeypatch, cli_main)

    with runner.isolated_filesystem():
        result = _invoke_build(runner, cli_main, allow_runtimes=(invalid_runtime,))

    assert result.exit_code == 2
    assert "Invalid --allow-runtime value(s)" in result.output
    assert "Expected format: module.ClassName" in result.output


def test_build_fails_when_model_implementation_format_is_invalid(
    monkeypatch, cli_main, runner
):
    _patch_build_pipeline(monkeypatch, cli_main)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {
                "models/custom/model-settings.json": {
                    "name": "custom-model",
                    "implementation": "invalid-runtime",
                }
            }
        )

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Invalid implementation" in result.output
    assert "Expected format: module.ClassName" in result.output


def test_build_fails_when_model_settings_json_is_not_an_object(
    monkeypatch, cli_main, runner
):
    _patch_build_pipeline(monkeypatch, cli_main)

    with runner.isolated_filesystem():
        _write_model_settings_batch(
            {"models/custom/model-settings.json": ["not-an-object"]}
        )

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Invalid JSON schema" in result.output
    assert "expected a JSON object" in result.output


def test_build_fails_when_model_settings_json_is_malformed(
    monkeypatch, cli_main, runner
):
    _patch_build_pipeline(monkeypatch, cli_main)

    with runner.isolated_filesystem():
        path = Path("models/custom/model-settings.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ invalid json }", encoding="utf-8")

        result = _invoke_build(runner, cli_main)

    assert result.exit_code == 2
    assert "Invalid JSON in" in result.output


def test_dockerfile_passes_allow_runtime_to_dockerfile_generator(
    monkeypatch, cli_main, runner
):
    captured = {}

    def _fake_generate_dockerfile(*args, **kwargs):
        captured["custom_runtimes"] = kwargs.get("custom_runtimes")
        return "FROM test"

    def _fake_write_dockerfile(folder, dockerfile, include_dockerignore):
        captured["folder"] = folder
        captured["dockerfile"] = dockerfile
        captured["include_dockerignore"] = include_dockerignore
        return "Dockerfile"

    monkeypatch.setattr(cli_main, "generate_dockerfile", _fake_generate_dockerfile)
    monkeypatch.setattr(cli_main, "write_dockerfile", _fake_write_dockerfile)

    with runner.isolated_filesystem():
        result = _invoke_dockerfile(
            runner,
            cli_main,
            allow_runtimes=("custom.Runtime", "another.Runtime"),
        )

    assert result.exit_code == 0
    assert result.exception is None
    assert captured["custom_runtimes"] == ["custom.Runtime", "another.Runtime"]
    assert captured["folder"] == "."
    assert captured["dockerfile"] == "FROM test"
