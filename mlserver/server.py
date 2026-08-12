import asyncio
import signal

from mlserver.repository.factory import ModelRepositoryFactory

from .model import MLModel
from .settings import Settings, ModelSettings, log_runtime_security_mode
from .logging import configure_logger
from .registry import MultiModelRegistry
from .handlers import DataPlane, ModelRepositoryHandlers
from .parallel import InferencePoolRegistry
from .batching import load_batching
from .rest import RESTServer
from .grpc import GRPCServer
from .metrics import MetricsServer
from .kafka import KafkaServer
from .utils import logger

HANDLED_SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]


class MLServer:
    """Top-level server orchestrator that wires together all transport servers,
    the model registry, parallel inference pools, and the data plane.

    On construction the server creates the REST, gRPC, Kafka (optional), and
    Metrics (optional) servers, a :class:`MultiModelRegistry` with lifecycle
    hooks, and — when ``parallel_workers`` is enabled — an
    :class:`InferencePoolRegistry` for GIL-free parallel inference.

    The :meth:`start` / :meth:`stop` pair controls the full server lifecycle.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._add_signal_handlers()

        self._metrics_server = None
        if self._settings.metrics_endpoint:
            self._metrics_server = MetricsServer(self._settings)

        self._inference_pool_registry = None
        if self._settings.parallel_workers:
            # Only load inference pool if parallel inference has been enabled
            on_worker_stop = []
            if self._metrics_server:
                on_worker_stop = [self._metrics_server.on_worker_stop]

            self._inference_pool_registry = InferencePoolRegistry(
                self._settings, on_worker_stop=on_worker_stop  # type: ignore
            )

        self._model_registry = self._create_model_registry()
        self._model_repository = ModelRepositoryFactory.resolve_model_repository(
            self._settings
        )
        self._data_plane = DataPlane(
            settings=self._settings, model_registry=self._model_registry
        )
        self._model_repository_handlers = ModelRepositoryHandlers(
            repository=self._model_repository, model_registry=self._model_registry
        )

        self._configure_logger()
        self._create_servers()

    def _create_model_registry(self) -> MultiModelRegistry:
        """Build a :class:`MultiModelRegistry` with the appropriate lifecycle
        hooks.  When parallel inference is enabled the pool registry's
        ``load_model`` / ``reload_model`` / ``unload_model`` hooks are prepended
        so that every model is transparently loaded into worker processes."""
        on_model_load = [
            self.add_custom_handlers,
            load_batching,
        ]
        on_model_reload = [self.reload_custom_handlers]
        on_model_unload = [self.remove_custom_handlers]

        if not self._inference_pool_registry:
            return MultiModelRegistry(
                on_model_load=on_model_load,  # type: ignore
                on_model_reload=on_model_reload,  # type: ignore
                on_model_unload=on_model_unload,  # type: ignore
            )

        on_model_load = [
            self._inference_pool_registry.load_model,
            self.add_custom_handlers,
            load_batching,
        ]
        on_model_reload = [
            self._inference_pool_registry.reload_model,  # type: ignore
            self.reload_custom_handlers,
        ]
        on_model_unload = [
            self._inference_pool_registry.unload_model,  # type: ignore
            self.remove_custom_handlers,
        ]

        return MultiModelRegistry(
            on_model_load=on_model_load,  # type: ignore
            on_model_reload=on_model_reload,  # type: ignore
            on_model_unload=on_model_unload,  # type: ignore
            model_initialiser=self._inference_pool_registry.model_initialiser,
        )

    def _configure_logger(self):
        self._logger = configure_logger(self._settings)

    def _create_servers(self):
        self._rest_server = RESTServer(
            self._settings, self._data_plane, self._model_repository_handlers
        )
        self._grpc_server = GRPCServer(
            self._settings, self._data_plane, self._model_repository_handlers
        )

        self._kafka_server = None
        if self._settings.kafka_enabled:
            self._kafka_server = KafkaServer(self._settings, self._data_plane)

    async def start(self, models_settings: list[ModelSettings] = []):
        """Start all transport servers and load the initial set of models.

        The startup sequence is:
        1. Validate runtime security configuration — abort if the trusted
           runtimes allowlist is invalid.
        2. Start REST, gRPC, Metrics, and Kafka servers concurrently.
        3. Load all ``models_settings`` concurrently.
        4. Mark ``startup_complete()`` so the readiness endpoint begins
           returning ``true``.

        If any model fails to load during startup the server is shut down
        gracefully.
        """
        # Validate runtime security configuration before starting servers to prevent
        # a window where endpoints are accessible but security hasn't been verified
        try:
            log_runtime_security_mode()
        except Exception as exc:
            logger.exception("Failed to load trusted runtimes allowlist!")
            raise RuntimeError(
                "Server startup aborted: "
                "invalid trusted runtimes allowlist configuration"
            ) from exc

        servers = [self._rest_server.start(), self._grpc_server.start()]
        if self._metrics_server:
            servers.append(self._metrics_server.start())

        if self._kafka_server:
            servers.append(self._kafka_server.start())

        servers_task = asyncio.gather(*servers)

        try:
            await asyncio.gather(
                *[
                    self._model_registry.load(model_settings)
                    for model_settings in models_settings
                ]
            )
            # Mark startup complete only if all models loaded successfully
            self._model_registry.startup_complete()
        except Exception:
            # If one of the models failed to load during startup, shutdown the
            # server gracefully
            logger.exception("Some of the models failed to load during startup!")
            await self.stop()
        finally:
            await servers_task

    async def add_custom_handlers(self, model: MLModel) -> MLModel:
        """Register any custom REST and Kafka endpoints exposed by *model*."""
        await self._rest_server.add_custom_handlers(model)
        if self._kafka_server:
            await self._kafka_server.add_custom_handlers(model)

        # TODO: Add support for custom gRPC endpoints
        # self._grpc_server.add_custom_handlers(handlers)

        return model

    async def reload_custom_handlers(self, old_model: MLModel, new_model: MLModel):
        """Replace custom handlers: register *new_model*'s handlers then remove
        *old_model*'s."""
        await self.add_custom_handlers(new_model)
        await self.remove_custom_handlers(old_model)

        # TODO: Add support for custom gRPC endpoints
        # self._grpc_server.delete_custom_handlers(handlers)

        return new_model

    async def remove_custom_handlers(self, model: MLModel):
        """Remove custom REST and Kafka endpoints previously registered by
        *model*."""
        await self._rest_server.delete_custom_handlers(model)
        if self._kafka_server:
            await self._kafka_server.delete_custom_handlers(model)

        # TODO: Add support for custom gRPC endpoints
        # self._grpc_server.delete_custom_handlers(handlers)

        return model

    def _add_signal_handlers(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running yet, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        for sig in HANDLED_SIGNALS:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(self.stop(sig=s))
            )

    async def stop(self, sig: int | None = None):
        """Gracefully shut down all server components in reverse-start order:
        inference pools → Kafka → gRPC → REST → Metrics."""
        if self._inference_pool_registry:
            await self._inference_pool_registry.close()

        if self._kafka_server:
            await self._kafka_server.stop()

        if self._grpc_server:
            await self._grpc_server.stop(sig)

        if self._rest_server:
            await self._rest_server.stop(sig)

        if self._metrics_server:
            await self._metrics_server.stop(sig)
