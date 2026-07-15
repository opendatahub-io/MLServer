import pytest

from mlserver.model import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.batching.hooks import (
    load_batching,
    unload_batching,
    reload_batching,
    _AdaptiveBatchingAttr,
)


async def test_batching_predict(
    sum_model: MLModel, inference_request: InferenceRequest
):
    await load_batching(sum_model)
    response = await sum_model.predict(inference_request)

    assert response is not None
    assert isinstance(response, InferenceResponse)
    assert len(response.outputs) == 1


async def test_batching_predict_stream(
    text_stream_model: MLModel, generate_request: InferenceRequest, caplog
):
    # Force batching to be enabled
    text_stream_model.settings.max_batch_size = 10
    text_stream_model.settings.max_batch_time = 0.4
    await load_batching(text_stream_model)

    async def get_stream_request(request):
        yield request

    stream = text_stream_model.predict_stream(get_stream_request(generate_request))
    responses = [r async for r in stream]

    assert len(responses) > 0
    assert isinstance(responses[0], InferenceResponse)
    assert len(responses[0].outputs) == 1
    assert "not supported for inference streaming" in caplog.records[0].message


@pytest.mark.parametrize(
    "max_batch_size, max_batch_time",
    [
        (0, 10),
        (-1, 10),
        (1, 10),
        (10, 0),
        (10, -1),
        (0, 0),
    ],
)
async def test_load_batching_disabled(
    max_batch_size: int, max_batch_time: float, sum_model: MLModel
):
    sum_model.settings.max_batch_size = max_batch_size
    sum_model.settings.max_batch_time = max_batch_time

    expected = sum_model.predict
    await load_batching(sum_model)

    assert expected == sum_model.predict  # type: ignore


async def test_unload_batching_removes_batcher(sum_model: MLModel):
    """Test that unload_batching removes the batcher attribute"""
    # Enable and load batching
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4
    await load_batching(sum_model)

    # Batcher should exist
    assert hasattr(sum_model, _AdaptiveBatchingAttr)

    # Unload batching
    await unload_batching(sum_model)

    # Batcher should be removed
    assert not hasattr(sum_model, _AdaptiveBatchingAttr)


async def test_unload_batching_idempotent(sum_model: MLModel):
    """Test that unload_batching is safe to call on model without batching"""
    # Don't load batching
    assert not hasattr(sum_model, _AdaptiveBatchingAttr)

    # Should not raise
    result = await unload_batching(sum_model)
    assert result is sum_model


async def test_reload_batching_cleans_up_old_and_sets_up_new(
    sum_model: MLModel, simple_model: MLModel
):
    """Test that reload_batching cleans up old model and sets up new model"""
    # Enable batching on old model
    sum_model.settings.max_batch_size = 10
    sum_model.settings.max_batch_time = 0.4

    # Load batching on old model
    await load_batching(sum_model)

    old_batcher = getattr(sum_model, _AdaptiveBatchingAttr)
    assert old_batcher is not None

    # Enable batching on new model
    simple_model.settings.max_batch_size = 10
    simple_model.settings.max_batch_time = 0.1

    # Reload batching
    result = await reload_batching(sum_model, simple_model)

    # Old model batcher should be removed
    assert not hasattr(sum_model, _AdaptiveBatchingAttr)

    # New model should have batching loaded
    assert hasattr(simple_model, _AdaptiveBatchingAttr)
    new_batcher = getattr(simple_model, _AdaptiveBatchingAttr)
    assert new_batcher is not old_batcher

    # Should return new model
    assert result is simple_model
