from fastapi import status

from ..errors import MLServerError


class InvalidModelContext(MLServerError):
    """Raised when a contextual method is called outside a model context."""

    def __init__(self):
        msg = (
            "Contextual method (e.g. mlserver.log() or mlserver.register())"
            " was called outside of a model context. "
        )
        super().__init__(msg, status.HTTP_500_INTERNAL_SERVER_ERROR)


class MetricNotFound(MLServerError):
    """Raised when a metric name is not registered or has an unexpected type."""

    def __init__(self, metric_name: str, collector=None):
        """Initialise with the metric name and optional mismatched collector."""
        msg = f"No metric found with name '{metric_name}'"
        if collector:
            msg = (
                f"Invalid metric found with name '{metric_name}'."
                f" Object is of type '{type(collector)}' instead of"
                " 'MetricWrapperBase'."
            )

        # NOTE: Most likely, this is an issue with the inference runtime's
        # code, therefore it makes sense to raise an internal `500` error (i.e.
        # instead of `404` - common for not found errors).
        super().__init__(msg, status.HTTP_500_INTERNAL_SERVER_ERROR)
