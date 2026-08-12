from typing import Any
from mlserver.codecs.base import InputCodec, register_input_codec
from mlserver.types import RequestInput, ResponseOutput, Parameters
from transformers.pipelines import Conversation
from mlserver.codecs.lists import is_list_of
from .utils import json_decode, json_encode


@register_input_codec
class HuggingfaceConversationCodec(InputCodec):
    """
    Codec that convers to / from a transformers Conversation input.
    """

    ContentType = "hg_conversation"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        """Return True if payload is a list of Conversation objects."""
        return is_list_of(payload, Conversation)

    @classmethod
    def encode_output(
        cls, name: str, payload: list[Conversation], use_bytes: bool = True, **kwargs
    ) -> ResponseOutput:
        """Encode a list of Conversations as a JSON BYTES ResponseOutput."""
        encoded = [json_encode(item, use_bytes=use_bytes) for item in payload]
        shape = [len(encoded), 1]
        return ResponseOutput(
            name=name,
            parameters=Parameters(
                content_type=cls.ContentType,
            ),
            datatype="BYTES",
            shape=shape,
            data=encoded,
        )

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> list[Any]:
        """Decode a BYTES ResponseOutput into Conversation-like dicts."""
        packed = response_output.data
        return [json_decode(item) for item in packed]

    @classmethod
    def encode_input(
        cls, name: str, payload: list[Conversation], use_bytes: bool = True, **kwargs
    ) -> RequestInput:
        """Encode a list of Conversations as a JSON BYTES RequestInput."""
        output = cls.encode_output(name, payload, use_bytes)
        return RequestInput(
            name=output.name,
            parameters=Parameters(
                content_type=cls.ContentType,
            ),
            datatype=output.datatype,
            shape=output.shape,
            data=output.data,
        )

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> list[Conversation]:
        """Decode a BYTES RequestInput into Conversation objects."""
        packed = request_input.data
        return [json_decode(item) for item in packed]
