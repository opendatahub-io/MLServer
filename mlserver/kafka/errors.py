from fastapi import status

from ..errors import MLServerError


class InvalidMessageHeaders(MLServerError):
    """Raised when a required header is missing from an incoming Kafka message."""

    def __init__(self, missing_header: str):
        """Build an error identifying the missing Kafka message header."""
        msg = (
            f"Invalid Kafka message. Expected '{missing_header}' header not "
            "found in message."
        )
        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)
