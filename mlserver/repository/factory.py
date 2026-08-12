from .repository import ModelRepository, SchemalessModelRepository
from ..settings import Settings
from pydantic import ImportString


class ModelRepositoryFactory:
    """Factory that instantiates the configured :class:`ModelRepository`
    implementation from server settings."""

    @staticmethod
    def resolve_model_repository(settings: Settings) -> ModelRepository:
        """Create the model repository specified in *settings*, defaulting
        to :class:`SchemalessModelRepository`."""
        model_repository_implementation: ImportString = SchemalessModelRepository

        result: ModelRepository
        if settings.model_repository_implementation:
            model_repository_implementation = settings.model_repository_implementation

        result = model_repository_implementation(
            root=settings.model_repository_root,
            **settings.model_repository_implementation_args,
        )

        return result
