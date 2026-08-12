import asyncio
import logging

import torch
import transformers
from huggingface_hub.utils import logging as hf_hub_logging

from mlserver.model import MLModel
from mlserver.settings import ModelSettings
from mlserver.logging import logger
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
)

from .settings import get_huggingface_settings
from .common import load_pipeline_from_settings
from .codecs import HuggingfaceRequestCodec
from .metadata import METADATA


class HuggingFaceRuntime(MLModel):
    """
    Implementation of the MLModel interface to load and serve Hugging Face
    Transformers pipelines.

    Supports any task available via ``transformers.pipeline`` (e.g.
    text-classification, fill-mask, image-classification). The task and
    model name are configured through ``HuggingFaceSettings``.
    """

    def __init__(self, settings: ModelSettings):
        """Initialise the runtime and resolve HuggingFace-specific settings."""
        self.hf_settings = get_huggingface_settings(settings)
        super().__init__(settings)

    def _configure_framework_logger(self) -> None:
        """Align transformers and huggingface_hub with MLServer's log level."""
        level = self._mlserver_log_level
        transformers.logging.set_verbosity(level)
        hf_hub_logging.set_verbosity(level)
        logger.debug(
            "Configured %s framework logger to %s",
            "huggingface",
            logging.getLevelName(level),
        )

    async def load(self) -> bool:
        """Load and cache the Hugging Face Transformers pipeline.

        Downloads the model on the first call (off the event loop to avoid
        blocking), then loads from the local cache on subsequent calls.
        """
        # Loading & caching pipeline in asyncio loop to avoid blocking
        logger.info(f"Loading model for task '{self.hf_settings.task_name}'...")
        await asyncio.get_running_loop().run_in_executor(
            None,
            load_pipeline_from_settings,
            self.hf_settings,
            self.settings,
        )

        # Now we load the cached model which should not block asyncio
        self._model = load_pipeline_from_settings(self.hf_settings, self.settings)
        self._merge_metadata()
        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Run the Transformers pipeline on the decoded request inputs."""
        # TODO: convert and validate?
        kwargs = HuggingfaceRequestCodec.decode_request(payload)
        args = kwargs.pop("args", [])

        array_inputs = kwargs.pop("array_inputs", [])
        if array_inputs:
            args = [list(array_inputs)] + args
        prediction = self._model(*args, **kwargs)

        return self.encode_response(
            payload=prediction, default_codec=HuggingfaceRequestCodec
        )

    async def unload(self) -> bool:
        """Release GPU memory held by the Transformers pipeline (PyTorch only)."""
        # TODO: Free up Tensorflow's GPU memory
        is_torch = self._model.framework == "pt"
        if not is_torch:
            return True

        uses_gpu = torch.cuda.is_available() and self._model.device != -1
        if not uses_gpu:
            # Nothing to free
            return True

        # Free up Torch's GPU memory
        torch.cuda.empty_cache()
        return True

    def _merge_metadata(self) -> None:
        meta = METADATA.get(self.hf_settings.task)
        if meta:
            self.inputs += meta.get("inputs", [])  # type: ignore
            self.outputs += meta.get("outputs", [])  # type: ignore
