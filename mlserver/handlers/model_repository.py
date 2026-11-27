import asyncio
from weakref import WeakValueDictionary

from ..utils import defer_cancellation, with_operation_lock
from ..settings import ModelSettings, DEFAULT_MODEL_OPERATION_TIMEOUT
from ..registry import MultiModelRegistry
from ..repository import ModelRepository
from ..errors import ModelNotFound, ModelUnloadError
from ..types import (
    RepositoryIndexRequest,
    RepositoryIndexResponse,
    RepositoryIndexResponseItem,
    State,
)

NO_VERSION_KEY = "__no_version__"


def _model_key(model_settings: ModelSettings) -> tuple[str, str]:
    return (model_settings.name, model_settings.version or "")


class ModelRepositoryHandlers:
    def __init__(
        self,
        repository: ModelRepository,
        model_registry: MultiModelRegistry,
        model_operation_timeout: float = DEFAULT_MODEL_OPERATION_TIMEOUT,
    ):
        self._operation_locks: WeakValueDictionary[str, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._repository = repository
        self._model_registry = model_registry
        self._model_operation_timeout = model_operation_timeout

    async def index(self, payload: RepositoryIndexRequest) -> RepositoryIndexResponse:
        # Get models from repository (on disk)
        repository_model_settings = await self._repository.list()

        self._model_registry.forget_missing_load_failures(repository_model_settings)

        if payload.include_registry_models:
            # Get all models from registry
            registry_model_settings = [
                m.settings for m in await self._model_registry.get_models()
            ]

            # Union and deduplicate models by (name, version)
            all_model_settings = self._union_model_settings(
                repository_model_settings, registry_model_settings
            )
        else:
            all_model_settings = repository_model_settings

        # Convert to index items
        index_items = []
        for model_settings in all_model_settings:
            index_item = await self._to_item(model_settings)
            if payload.ready:
                if index_item.state == State.READY:
                    index_items.append(index_item)
            else:
                index_items.append(index_item)

        return RepositoryIndexResponse(root=index_items)

    def _union_model_settings(
        self,
        repository_model_settings: list[ModelSettings],
        registry_model_settings: list[ModelSettings],
    ) -> list[ModelSettings]:
        """
        Union and deduplicate model settings by (name, version).

        Combines models from repository and registry, removing duplicates based on
        (name, version) key. When the same (name, version) appears in both sources,
        registry settings take precedence (shows what's actually running).

        Returns:
            Deduplicated list of model settings.
        """
        union_model_settings = {}
        # Repository first, then registry overwrites
        for model_settings in repository_model_settings + registry_model_settings:
            union_model_settings[_model_key(model_settings)] = model_settings

        return list(union_model_settings.values())

    async def _to_item(
        self, model_settings: ModelSettings
    ) -> RepositoryIndexResponseItem:
        version = model_settings.version
        try:
            model = await self._model_registry.get_model(model_settings.name, version)
        except ModelNotFound:
            model = None

        reason = ""
        # A versionless inference lookup can return a different, default version.
        # Each repository row describes only its exact artifact.
        if model is None or model.version != version:
            state = State.UNAVAILABLE
            if self._model_registry.has_load_failure(model_settings):
                reason = (
                    "Model failed to load. See the load response and server logs "
                    "for details."
                )
        elif not model.ready:
            state = State.LOADING
        elif not model._is_serving:
            state = State.UNAVAILABLE
            reason = "Model is loaded but has no eligible worker."
        else:
            state = State.READY
        return RepositoryIndexResponseItem(
            name=model_settings.name, version=version, state=state, reason=reason
        )

    def _get_operation_lock(self, name: str) -> asyncio.Lock:
        return self._operation_locks.setdefault(name, asyncio.Lock())

    @with_operation_lock(lambda self, name: self._get_operation_lock(name))
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def load(self, name: str) -> bool:
        try:
            all_model_settings = await self._repository.find(name)
        except ModelNotFound:
            self._model_registry.forget_missing_load_failures([], name=name)
            raise
        self._model_registry.forget_missing_load_failures(all_model_settings, name=name)

        loaded_versions = set()
        for model_settings in all_model_settings:
            model = await self._model_registry.load(model_settings)

            # Add to loaded versions set to later remove stale models
            model_version = model.version if model.version else NO_VERSION_KEY
            loaded_versions.add(model_version)

        # Remove stale models
        all_models = await self._model_registry.get_models(name)

        # Cleanup stale versions in parallel
        stale_versions = []
        for model in all_models:
            model_version = model.version if model.version else NO_VERSION_KEY
            if model_version not in loaded_versions:
                stale_versions.append((model.name, model.version))

        # Unload all stale versions in parallel (best-effort)
        if stale_versions:
            results = await asyncio.gather(
                *[
                    self._model_registry.unload_version(model_name, model_version)
                    for model_name, model_version in stale_versions
                ],
                return_exceptions=True,
            )
            # Check for failures and raise if any occurred
            unload_failures = [
                result for result in results if isinstance(result, Exception)
            ]
            if unload_failures:
                raise ModelUnloadError(
                    f"Failed to cleanup {len(unload_failures)} of "
                    f"{len(stale_versions)} stale version(s) of model {name} "
                    f"during repository load sync. Stale versions removed "
                    "from registry, but some resources "
                    "may still be held."
                )

        return True

    @with_operation_lock(lambda self, name: self._get_operation_lock(name))
    @defer_cancellation(lambda self, *args, **kwargs: self._model_operation_timeout)
    async def unload(self, name: str) -> bool:
        await self._model_registry.unload(name)

        return True
