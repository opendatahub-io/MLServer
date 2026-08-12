from typing import Any
from mlserver.types import RequestInput, ResponseOutput, Parameters
from mlserver.codecs.base import InputCodec, register_input_codec


@register_input_codec
class RawCodec(InputCodec):
    """
    Encode/Decode raw python datatypes
    """

    ContentType = "raw"

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        """Return True if payload is an int, str, or float."""
        return (
            isinstance(payload, int)
            or isinstance(payload, str)
            or isinstance(payload, float)
        )

    @classmethod
    def encode_output(
        cls, name: str, payload: int | str | float, **kwargs
    ) -> ResponseOutput:
        """Encode a single scalar value as a BYTES ResponseOutput."""
        return ResponseOutput(
            name=name,
            datatype="BYTES",
            shape=[1],
            data=[payload],
            parameters=Parameters(content_type=cls.ContentType),
        )

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> int | str | float:
        """Decode a ResponseOutput into a single scalar value."""
        return cls.decode_input(response_output)  # type: ignore

    @classmethod
    def encode_input(
        cls, name: str, payload: int | str | float, **kwargs
    ) -> RequestInput:
        """Encode a single scalar value as a BYTES RequestInput."""
        output = cls.encode_output(name=name, payload=payload)

        return RequestInput(
            name=output.name,
            datatype=output.datatype,
            shape=output.shape,
            data=output.data,
            parameters=output.parameters,
        )

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> int | str | float:
        """Decode a RequestInput into a single scalar value."""
        return request_input.data[0]
