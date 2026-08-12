from .settings import ModelSettings
from .types import InferenceRequest, InferenceResponse


class InferenceMiddleware:
    """
    Base class to implement middlewares.
    """

    def request_middleware(
        self, request: InferenceRequest, model_settings: ModelSettings
    ) -> InferenceRequest:
        """Transform an inference request before it reaches the model."""
        raise NotImplementedError()

    def response_middleware(
        self, response: InferenceResponse, model_settings: ModelSettings
    ) -> InferenceResponse:
        """Transform an inference response before it is returned to the caller."""
        raise NotImplementedError()


class InferenceMiddlewares(InferenceMiddleware):
    """Composite middleware that chains a list of :class:`InferenceMiddleware`
    instances.  Request middlewares are applied in order; response middlewares
    are applied in the same order (not reversed).
    """

    def __init__(self, *inference_middlewares):
        """Initialise with one or more :class:`InferenceMiddleware` instances."""
        self._middlewares = inference_middlewares

    def request_middleware(
        self, request: InferenceRequest, model_settings: ModelSettings
    ) -> InferenceRequest:
        """Apply all registered request middlewares in order."""
        processed_request = request
        for middleware in self._middlewares:
            processed_request = middleware.request_middleware(
                processed_request, model_settings
            )

        return processed_request

    def response_middleware(
        self, response: InferenceResponse, model_settings: ModelSettings
    ) -> InferenceResponse:
        """Apply all registered response middlewares in order."""
        processed_response = response
        for middleware in self._middlewares:
            processed_response = middleware.response_middleware(
                processed_response, model_settings
            )

        return processed_response
