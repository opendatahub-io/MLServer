from mlserver.errors import MLServerError


class RemoteInferenceError(MLServerError):
    """Raised when a remote V2 inference call returns a non-200 status."""

    def __init__(self, code: int, reason: str):
        super().__init__(f"Remote inference call failed with {code}, {reason}")


class InvalidExplanationShape(MLServerError):
    """Raised when an explanation response contains an unexpected number of elements."""

    def __init__(self, shape: list[int] | int):
        super().__init__(
            f"Expected a single element, but multiple were returned {shape}"
        )
