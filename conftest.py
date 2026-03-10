"""
Root test configuration shared across both `tests/` and `runtimes/*` suites.

This file must live at repository root so pytest applies the trusted-runtimes
fixture to runtime-specific tests invoked from tox (e.g. `runtimes/sklearn`),
which do not inherit from `tests/conftest.py`.
"""

import json
import os

import pytest

import mlserver.settings as mlserver_settings


TEST_ONLY_EXTRA_IMPLEMENTATIONS = {
    # Core repo test fixtures.
    "tests.fixtures.SumModel",
    "tests.fixtures.SlowModel",
    "tests.fixtures.SimpleModel",
    "tests.fixtures.ErrorModel",
    "tests.fixtures.EnvModel",
    "tests.fixtures.EchoModel",
    "tests.fixtures.TextModel",
    "tests.fixtures.TextStreamModel",
    "tests.metrics.test_custom.CustomMetricsModel",
    "fixtures.SumModel",
    "custom.SumModel",
    "env_models.DummySKLearnModel",
}


def _configure_test_trusted_runtimes_artifact() -> None:
    test_allowed_model_implementations = (
        mlserver_settings.ALLOWED_MODEL_IMPLEMENTATIONS.union(
            TEST_ONLY_EXTRA_IMPLEMENTATIONS
        )
    )

    artifact_dir = os.path.join(
        os.path.dirname(__file__), ".pytest_cache", "trusted-runtimes"
    )
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "trusted-runtimes.json")
    with open(artifact_path, "w", encoding="utf-8") as artifact_file:
        artifact_file.write(
            json.dumps(sorted(test_allowed_model_implementations)),
        )

    os.environ[mlserver_settings.INTERNAL_TEST_HOOKS_ENV] = "1"
    os.environ[mlserver_settings.INTERNAL_TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV] = (
        artifact_path
    )
    mlserver_settings._get_allowed_model_implementations.cache_clear()
    mlserver_settings._load_image_baked_allowed_model_implementations.cache_clear()


def _cleanup_test_trusted_runtimes_artifact() -> None:
    os.environ.pop(mlserver_settings.INTERNAL_TEST_HOOKS_ENV, None)
    os.environ.pop(
        mlserver_settings.INTERNAL_TEST_TRUSTED_RUNTIMES_ARTIFACT_ENV,
        None,
    )
    mlserver_settings._get_allowed_model_implementations.cache_clear()
    mlserver_settings._load_image_baked_allowed_model_implementations.cache_clear()


def pytest_configure(config: pytest.Config) -> None:
    # Run before test collection so import-time ModelSettings validations can use
    # the test trusted-runtimes artifact.
    # Ownership note: this root hook configures trusted-runtime state globally
    # for both `tests/` and `runtimes/*` suites.
    _configure_test_trusted_runtimes_artifact()


def pytest_unconfigure(config: pytest.Config) -> None:
    _cleanup_test_trusted_runtimes_artifact()
