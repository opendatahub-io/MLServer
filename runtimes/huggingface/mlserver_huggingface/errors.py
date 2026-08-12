from mlserver.errors import MLServerError


class MissingHuggingFaceSettings(MLServerError):
    """Raised when no HuggingFace settings are provided via env or model config."""

    def __init__(self):
        super().__init__("Missing HuggingFace Runtime settings.")


class InvalidTransformersTask(MLServerError):
    """Raised when the requested task is not supported by Transformers."""

    def __init__(self, task: str, available_tasks: list[str]):
        msg = f"Invalid transformer task: {task}. Available tasks: {available_tasks}."
        super().__init__(msg)


class InvalidOptimumTask(MLServerError):
    """Raised when the requested task is not supported by Optimum ONNX Runtime."""

    def __init__(self, task: str, available_tasks: list[str]):
        msg = (
            "Invalid transformer task for Optimum model: {task}. "
            f"Available Optimum tasks: {available_tasks}."
        )
        super().__init__(msg)


class InvalidModelParameter(MLServerError):
    """Raised when a model parameter value cannot be parsed as its declared type."""

    def __init__(self, name: str, value: str, param_type: str):
        msg = (
            f"Bad model parameter: {name}"
            f" with value {value}"
            f" can't be parsed as a {param_type}"
        )
        super().__init__(msg)


class InvalidModelParameterType(MLServerError):
    """Raised when a model parameter declares an unsupported type identifier."""

    def __init__(self, param_type: str):
        msg = (
            f"Bad model parameter type: {param_type}."
            f" Only valid types are INT, FLOAT, DOUBLE, STRING, BOOL."
        )
        super().__init__(msg)
