"""V2 inference and model-management calls used by the mixed-model benchmark."""

from collections.abc import Sequence
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiohttp
import grpc
import orjson

from mlserver.grpc import dataplane_pb2 as pb
from mlserver.grpc.converters import ModelInferRequestConverter
from mlserver.grpc.dataplane_pb2_grpc import GRPCInferenceServiceStub

if TYPE_CHECKING:
    from .run import Target


class GrpcCalls:
    protocol = "grpc"

    def __init__(self, client: GRPCInferenceServiceStub, models: Sequence["Target"]):
        self.client = client
        self.templates = [
            ModelInferRequestConverter.from_types(
                m.request,
                m.name,
                m.version,
                # The V2 raw encoder converts JSON strings to length-prefixed UTF-8.
                use_raw=any(inp.datatype == "BYTES" for inp in m.request.inputs),
            )
            for m in models
        ]

    async def infer(
        self, index: int, request_id: str, timeout: float
    ) -> pb.ModelInferResponse:
        request = pb.ModelInferRequest()
        request.CopyFrom(self.templates[index])
        request.id = request_id
        return await self.client.ModelInfer(request, timeout=timeout)

    async def manage(self, operation: str, model: str, timeout: float) -> None:
        if operation == "load":
            await self.client.RepositoryModelLoad(
                pb.RepositoryModelLoadRequest(model_name=model), timeout=timeout
            )
        else:
            await self.client.RepositoryModelUnload(
                pb.RepositoryModelUnloadRequest(model_name=model), timeout=timeout
            )


class RestCalls:
    protocol = "rest"

    def __init__(self, client: aiohttp.ClientSession, models: Sequence["Target"]):
        self.client = client
        self.paths = [
            f"/v2/models/{quote(m.name, safe='')}"
            f"/versions/{quote(m.version, safe='')}/infer"
            for m in models
        ]
        self.templates = [m.request.model_dump(mode="json") for m in models]

    async def _post(self, path: str, data: bytes, timeout: float) -> bytes:
        async with self.client.post(
            path,
            data=data,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=timeout, ceil_threshold=float("inf")),
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                raise aiohttp.ClientResponseError(
                    response.request_info, response.history, status=response.status
                )
            return await response.read()

    async def infer(self, index: int, request_id: str, timeout: float) -> bytes:
        body = orjson.dumps({**self.templates[index], "id": request_id})
        return await self._post(self.paths[index], body, timeout)

    async def manage(self, operation: str, model: str, timeout: float) -> None:
        await self._post(
            f"/v2/repository/models/{quote(model, safe='')}/{operation}", b"{}", timeout
        )


def failure(
    exc: grpc.aio.AioRpcError | aiohttp.ClientError | TimeoutError,
) -> tuple[str, str]:
    if isinstance(exc, grpc.aio.AioRpcError):
        code = exc.code()
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return "deadline", code.name
        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            reason = exc.details() or ""
            outcome = (
                "client_receive_limit"
                if reason.startswith("CLIENT: Received message larger than max")
                else "rejected"
            )
            return outcome, f"{code.name}: {reason}" if reason else code.name
        return "rpc_error", code.name
    if isinstance(exc, TimeoutError):
        return "deadline", "CLIENT_TIMEOUT"
    if isinstance(exc, aiohttp.ClientResponseError):
        return ("rejected" if exc.status == 429 else "http_error"), f"HTTP_{exc.status}"
    return "http_error", type(exc).__name__
