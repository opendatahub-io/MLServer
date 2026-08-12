import json

from asyncio import CancelledError
from enum import IntEnum
from pydantic import BaseModel, Field, ConfigDict
from typing import Any

from ..utils import generate_uuid
from ..settings import ModelSettings


class ModelUpdateType(IntEnum):
    """Type of model lifecycle update dispatched to workers."""

    Load = 1
    Unload = 2


class Message(BaseModel):
    """Base message exchanged between the main process and inference workers."""

    model_config = ConfigDict(
        protected_namespaces=(),
    )

    id: str = Field(default_factory=generate_uuid)


class ModelRequestMessage(Message):
    """Request sent to a worker to invoke a model method (e.g. predict)."""

    model_name: str
    model_version: str | None = None
    method_name: str
    method_args: list[Any] = []
    method_kwargs: dict[str, Any] = {}


class ModelResponseMessage(Message):
    """Response from a worker carrying a return value or an exception."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    return_value: Any | None = None
    exception: Exception | CancelledError | None = None


class ModelUpdateMessage(Message):
    """Instructs workers to load or unload a model.

    Accepts a ``model_settings`` kwarg at init which is serialised to JSON
    for safe cross-process transfer.
    """

    update_type: ModelUpdateType
    serialised_model_settings: str

    def __init__(self, *args, **kwargs):
        """Serialise *model_settings* to JSON for cross-process transfer."""
        model_settings = kwargs.pop("model_settings", None)
        if model_settings:
            as_dict = model_settings.model_dump(by_alias=True)
            # Ensure the private `_source` attr also gets serialised
            if model_settings._source:
                as_dict["_source"] = model_settings._source

            kwargs["serialised_model_settings"] = json.dumps(as_dict)

        return super().__init__(*args, **kwargs)

    @property
    def model_settings(self) -> ModelSettings:
        """Deserialise and return the model settings from JSON."""
        return ModelSettings.parse_raw(self.serialised_model_settings)
