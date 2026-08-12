from fastapi import status


class MLServerError(Exception):
    """Base exception for all MLServer errors, carrying an HTTP status code."""

    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        """Initialise with an error message and optional HTTP status code."""
        super().__init__(msg)
        self.status_code = status_code


class InvalidModelURI(MLServerError):
    """Raised when a model's URI is missing or malformed (HTTP 422)."""

    def __init__(self, name: str, model_uri: str | None = None):
        """Initialise with the model name and its optional URI string."""
        msg = f"Invalid URI specified for model {name}"
        if model_uri:
            msg += f" ({model_uri})"

        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ModelNotFound(MLServerError):
    """Raised when a requested model (and optional version) does not exist (HTTP 404)."""

    def __init__(self, name: str, version: str | None = None):
        """Initialise with the model name and optional version."""
        msg = f"Model {name} not found"
        if version is not None:
            msg = f"Model {name} with version {version} not found"

        super().__init__(msg, status.HTTP_404_NOT_FOUND)


class ModelNotReady(MLServerError):
    """Raised when a model exists but has not finished loading yet."""

    def __init__(self, name: str, version: str | None = None):
        """Initialise with the model name and optional version."""
        msg = f"Model {name} is not ready yet."
        if version is not None:
            msg = f"Model {name} with version {version} is not ready yet."

        super().__init__(msg, status.HTTP_400_BAD_REQUEST)


class ModelLoadError(MLServerError):
    """Raised when model loading or validation fails (e.g. invalid artifact)."""

    def __init__(self, msg: str):
        """Initialise with a descriptive error message."""
        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


class ModelValidationError(MLServerError):
    """Raised when model configuration or options are invalid (e.g. at load time)."""

    def __init__(self, msg: str):
        """Initialise with a descriptive error message."""
        super().__init__(msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


class InferenceError(MLServerError):
    """Raised when an inference request fails during prediction."""

    def __init__(self, msg: str):
        """Initialise with a descriptive error message."""
        super().__init__(msg, status.HTTP_400_BAD_REQUEST)


class ModelParametersMissing(MLServerError):
    """Raised when required parameters are not provided for a model."""

    def __init__(self, model_name: str):
        """Initialise with the name of the model missing parameters."""
        super().__init__(
            f"Parameters missing for model {model_name}", status.HTTP_400_BAD_REQUEST
        )
