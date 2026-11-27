"""Public readiness after a terminal failure of one pool's response receiver."""

import asyncio

import grpc
import pytest
from httpx import AsyncClient

from mlserver.grpc import dataplane_pb2 as pb
from mlserver.grpc.converters import ModelInferRequestConverter
from mlserver.grpc.dataplane_pb2_grpc import GRPCInferenceServiceStub

zmq = pytest.importorskip("zmq")


@pytest.fixture
def settings(settings):
    settings.parallel_workers = 1
    settings.parallel_inference_transport = "zmq"
    settings.strict_readiness = True
    return settings


async def test_terminal_receiver_withdraws_only_affected_pool(
    mlserver, rest_client, settings, sum_model_settings, inference_request
):
    async with asyncio.timeout(30):
        await rest_client.wait_until_ready()
        healthy = sum_model_settings
        affected = healthy.model_copy(deep=True, update={"name": "affected"})
        affected.parameters.inference_pool_gid = "isolated"
        registry = mlserver._model_registry
        await registry.load(affected)
        dispatcher = mlserver._inference_pool_registry._pools["isolated"]._dispatcher
        transport = dispatcher._transport
        workers = dispatcher._workers.copy()

        async with (
            AsyncClient(
                base_url=f"http://{settings.host}:{settings.http_port}"
            ) as http,
            grpc.aio.insecure_channel(
                f"{settings.host}:{settings.grpc_port}"
            ) as channel,
        ):
            rpc = GRPCInferenceServiceStub(channel)

            async def check_model(model, ready):
                url = f"/v2/models/{model.name}/versions/{model.version}"
                probe = pb.ModelReadyRequest(name=model.name, version=model.version)
                assert (await rpc.ModelReady(probe)).ready is ready
                assert (await http.get(url + "/ready")).status_code == (
                    200 if ready else 400
                )
                infer = ModelInferRequestConverter.from_types(
                    inference_request, model.name, model.version
                )
                response = await http.post(
                    url + "/infer", json=inference_request.model_dump()
                )
                if ready:
                    assert response.status_code == 200
                    assert response.json()["outputs"][0]["data"] == [6]
                    reply = await rpc.ModelInfer(infer)
                    assert (reply.model_name, reply.model_version) == (
                        model.name,
                        model.version,
                    )
                    assert list(reply.outputs[0].contents.int64_contents) == [6]
                else:
                    assert response.status_code == 500
                    with pytest.raises(grpc.aio.AioRpcError) as error:
                        await rpc.ModelInfer(infer)
                    assert error.value.code() == grpc.StatusCode.INTERNAL
                    assert (
                        "ZeroMQ response transport is unavailable"
                        in error.value.details()
                    )

            async def check_server(ready, names):
                assert (await rpc.ServerReady(pb.ServerReadyRequest())).ready is ready
                assert (await http.get("/v2/health/ready")).status_code == (
                    200 if ready else 400
                )
                payload = {"ready": True, "include_registry_models": True}
                rows = (await http.post("/v2/repository/index", json=payload)).json()
                assert {row["name"] for row in rows} == names
                index = await rpc.RepositoryIndex(pb.RepositoryIndexRequest(**payload))
                assert {row.name for row in index.models} == names

            await check_model(affected, True)
            await check_model(healthy, True)
            await check_server(True, {affected.name, healthy.name})
            with zmq.asyncio.Context() as context, context.socket(zmq.PUSH) as peer:
                peer.connect(transport._reply_endpoint)
                await peer.send(b"invalid response")
                await asyncio.shield(transport._receiver)
            assert transport._failure is not None
            await check_model(affected, False)
            await check_model(healthy, True)
            await check_server(False, {healthy.name})
            settings.strict_readiness = False
            await check_server(True, {healthy.name})
            assert dispatcher._workers == workers
            assert all(worker.is_alive() for worker in workers.values())

            # An explicit unload retires the failed named pool. Reloading starts
            # a new receiver and confirms a new worker before readiness returns.
            await registry.unload(affected.name)
            await registry.load(affected)
            settings.strict_readiness = True
            await check_model(affected, True)
            await check_server(True, {affected.name, healthy.name})
