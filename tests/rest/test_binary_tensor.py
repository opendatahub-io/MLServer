"""Tests for the KServe v2 binary tensor data extension.

Unit tests verify parsing and encoding correctness.  Integration tests
use the same httpx/ASGI pattern as the existing REST endpoint tests.
"""

import numpy as np
import orjson
import pytest

from mlserver.rest.binary import (
    _encode_output_bytes,
    BinaryTensorError,
    DATATYPE_MAP,
    encode_binary_response,
    get_header_length,
    is_binary_request,
    parse_binary_request,
    resolve_binary_outputs,
)
from mlserver.codecs.numpy import NumpyCodec
from mlserver.codecs.string import StringCodec
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    Parameters,
    RequestInput,
    RequestOutput,
    ResponseOutput,
    TensorData,
)


@pytest.fixture
def fp32_tensor():
    """A [1, 3, 512, 512] FP32 tensor matching RF-DETR's input."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1, 3, 512, 512)).astype(np.float32)


@pytest.fixture
def binary_body(fp32_tensor):
    """A valid binary tensor request body for RF-DETR."""
    header = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 3, 512, 512],
                "datatype": "FP32",
                "parameters": {"binary_data_size": fp32_tensor.nbytes},
            }
        ],
        "outputs": [
            {"name": "dets", "parameters": {"binary_data": True}},
            {"name": "labels", "parameters": {"binary_data": True}},
        ],
    }
    header_bytes = orjson.dumps(header)
    return header_bytes + fp32_tensor.tobytes(), len(header_bytes)


def _rfdet_binary_body():
    """A [1, 3, 512, 512] FP32 binary body (the RF-DETR input shape)."""
    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((1, 3, 512, 512)).astype(np.float32)
    header = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 3, 512, 512],
                "datatype": "FP32",
                "parameters": {"binary_data_size": tensor.nbytes},
            }
        ]
    }
    header_bytes = orjson.dumps(header)
    return header_bytes + tensor.tobytes(), len(header_bytes)


def _unwrap(data):
    """Unwrap TensorData to the underlying payload."""
    return getattr(data, "root", data)


class TestIsBinaryRequest:
    def test_with_header(self):
        assert is_binary_request({"inference-header-content-length": "100"})

    def test_without_header(self):
        assert not is_binary_request({"content-type": "application/json"})

    def test_starlette_normalised(self):
        """Starlette normalises header names to lowercase."""
        assert is_binary_request({"inference-header-content-length": "50"})

    def test_empty_headers(self):
        assert not is_binary_request({})


class TestGetHeaderLength:
    def test_valid(self):
        assert get_header_length({"inference-header-content-length": "225"}) == 225

    def test_zero(self):
        assert get_header_length({"inference-header-content-length": "0"}) == 0

    def test_missing(self):
        with pytest.raises(BinaryTensorError, match="Missing"):
            get_header_length({})

    def test_non_integer(self):
        with pytest.raises(BinaryTensorError, match="not an integer"):
            get_header_length({"inference-header-content-length": "abc"})

    def test_negative(self):
        with pytest.raises(BinaryTensorError, match="negative"):
            get_header_length({"inference-header-content-length": "-1"})


class TestParseBinaryRequest:
    def test_valid_fp32(self, binary_body):
        body, header_length = binary_body
        request, header = parse_binary_request(body, header_length)

        assert isinstance(request, InferenceRequest)
        assert len(request.inputs) == 1
        assert request.inputs[0].name == "input"
        assert request.inputs[0].datatype == "FP32"
        assert request.inputs[0].shape == [1, 3, 512, 512]
        assert header["inputs"][0]["name"] == "input"

        data = _unwrap(request.inputs[0].data)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32
        # TensorData is flat even when the logical tensor has shape
        # (mirrors the gRPC raw-input representation).
        assert data.shape == (1 * 3 * 512 * 512,)
        assert data.size == np.prod(request.inputs[0].shape)

    def test_zero_copy(self, binary_body):
        """Verify np.frombuffer shares memory with the body."""
        body, header_length = binary_body
        request, _ = parse_binary_request(body, header_length)
        data = _unwrap(request.inputs[0].data)
        assert isinstance(data, np.ndarray)
        assert not data.flags["WRITEABLE"]  # frombuffer is read-only

    def test_header_exceeds_body(self):
        with pytest.raises(BinaryTensorError, match="exceeds body size"):
            parse_binary_request(b"short", 100)

    def test_invalid_json_header(self):
        with pytest.raises(BinaryTensorError, match="Invalid JSON"):
            parse_binary_request(b"not json{{", 8)

    def test_missing_inputs(self):
        header = orjson.dumps({"outputs": []})
        with pytest.raises(BinaryTensorError, match="missing 'inputs'"):
            parse_binary_request(header, len(header))

    def test_both_data_and_binary_size(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [2],
                        "datatype": "FP32",
                        "data": [1.0, 2.0],
                        "parameters": {"binary_data_size": 8},
                    }
                ]
            }
        )
        body = header + b"\x00" * 8
        with pytest.raises(BinaryTensorError, match="mutually exclusive"):
            parse_binary_request(body, len(header))

    def test_binary_size_exceeds_remaining(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [4],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": 1000},
                    }
                ]
            }
        )
        body = header + b"\x00" * 16
        with pytest.raises(BinaryTensorError, match="exceeds remaining"):
            parse_binary_request(body, len(header))

    def test_size_shape_mismatch(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [4],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": 32},
                    }
                ]
            }
        )
        body = header + b"\x00" * 32
        with pytest.raises(BinaryTensorError, match="doesn't match shape"):
            parse_binary_request(body, len(header))

    def test_trailing_bytes(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [2],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": 8},
                    }
                ]
            }
        )
        body = header + b"\x00" * 8 + b"extra"
        with pytest.raises(BinaryTensorError, match="unaccounted"):
            parse_binary_request(body, len(header))

    def test_unsupported_datatype(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "x",
                        "shape": [1],
                        "datatype": "STRING",
                        "parameters": {"binary_data_size": 4},
                    }
                ]
            }
        )
        body = header + b"\x00" * 4
        with pytest.raises(BinaryTensorError, match="unsupported datatype"):
            parse_binary_request(body, len(header))

    def test_multiple_inputs(self):
        tensor1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "a",
                        "shape": [3],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": tensor1.nbytes},
                    },
                    {
                        "name": "b",
                        "shape": [2, 2],
                        "datatype": "INT64",
                        "parameters": {"binary_data_size": tensor2.nbytes},
                    },
                ]
            }
        )
        body = header + tensor1.tobytes() + tensor2.tobytes()
        request, _ = parse_binary_request(body, len(header))

        assert np.array_equal(_unwrap(request.inputs[0].data), tensor1)
        assert np.array_equal(_unwrap(request.inputs[1].data), tensor2.reshape(-1))

    def test_mixed_binary_and_json_inputs(self):
        tensor = np.array([1.0, 2.0], dtype=np.float32)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "binary_input",
                        "shape": [2],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": tensor.nbytes},
                    },
                    {
                        "name": "json_input",
                        "shape": [2],
                        "datatype": "FP32",
                        "data": [3.0, 4.0],
                    },
                ]
            }
        )
        body = header + tensor.tobytes()
        request, _ = parse_binary_request(body, len(header))

        assert np.array_equal(_unwrap(request.inputs[0].data), tensor)
        assert list(_unwrap(request.inputs[1].data)) == [3.0, 4.0]


class TestDatatypeMap:
    def test_all_kserve_types_covered(self):
        expected = {
            "BOOL",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "INT8",
            "INT16",
            "INT32",
            "INT64",
            "FP16",
            "FP32",
            "FP64",
        }
        assert expected == set(DATATYPE_MAP.keys())

    def test_multi_byte_little_endian(self):
        for name, dtype in DATATYPE_MAP.items():
            if dtype.itemsize > 1:
                assert dtype.byteorder in (
                    "<",
                    "=",
                ), f"{name} not little-endian compatible: {dtype.byteorder}"


class TestEncodeBinaryResponse:
    def test_roundtrip(self):
        dets = np.random.default_rng(42).uniform(0, 1, (1, 300, 4)).astype(np.float32)
        labels = (
            np.random.default_rng(43).uniform(0, 1, (1, 300, 91)).astype(np.float32)
        )

        response = InferenceResponse(
            model_name="test",
            model_version="1",
            id="req-1",
            outputs=[
                ResponseOutput(
                    name="dets", datatype="FP32", shape=[1, 300, 4], data=dets
                ),
                ResponseOutput(
                    name="labels", datatype="FP32", shape=[1, 300, 91], data=labels
                ),
            ],
        )

        body, header_length = encode_binary_response(response, default_binary=True)

        header = orjson.loads(body[:header_length])
        assert header["model_name"] == "test"
        assert len(header["outputs"]) == 2
        assert header["outputs"][0]["parameters"]["binary_data_size"] == dets.nbytes
        assert header["outputs"][1]["parameters"]["binary_data_size"] == labels.nbytes

        offset = header_length
        dets_restored = np.frombuffer(
            body[offset : offset + dets.nbytes], dtype="<f4"
        ).reshape(1, 300, 4)
        offset += dets.nbytes
        labels_restored = np.frombuffer(
            body[offset : offset + labels.nbytes], dtype="<f4"
        ).reshape(1, 300, 91)

        np.testing.assert_array_equal(dets_restored, dets)
        np.testing.assert_array_equal(labels_restored, labels)

    def test_header_length_positive(self):
        response = InferenceResponse(
            model_name="test",
            outputs=[
                ResponseOutput(
                    name="x",
                    datatype="FP32",
                    shape=[1],
                    data=np.array([1.0], dtype=np.float32),
                ),
            ],
        )
        body, header_length = encode_binary_response(response, default_binary=True)
        assert header_length > 0
        assert len(body) > header_length

    def test_no_parameters_field_when_empty(self):
        response = InferenceResponse(
            model_name="test",
            outputs=[
                ResponseOutput(
                    name="x",
                    datatype="FP32",
                    shape=[1],
                    data=np.array([1.0], dtype=np.float32),
                ),
            ],
        )
        body, header_length = encode_binary_response(response, default_binary=True)
        header = orjson.loads(body[:header_length])
        assert "parameters" not in header


class TestResolveBinaryOutputs:
    """Per-output negotiation: request-wide default + named overrides."""

    def _request(self, outputs=None, parameters=None) -> InferenceRequest:
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name="in", datatype="FP32", shape=[1], data=TensorData([1.0])
                )
            ],
            outputs=outputs,
            parameters=parameters,
        )

    def test_no_outputs_uses_request_default(self):
        default, per_output = resolve_binary_outputs(self._request())
        assert default is False
        assert per_output is None

    def test_no_outputs_request_default_true(self):
        request = self._request(parameters=Parameters(binary_data_output=True))
        default, per_output = resolve_binary_outputs(request)
        assert default is True
        assert per_output is None

    def test_per_output_override_true(self):
        request = self._request(
            outputs=[
                RequestOutput(name="x", parameters=Parameters(binary_data=True)),
                RequestOutput(name="y"),
            ]
        )
        default, per_output = resolve_binary_outputs(request)
        assert default is False
        assert per_output == {"x": True, "y": False}

    def test_per_output_override_false_against_default_true(self):
        request = self._request(
            outputs=[
                RequestOutput(name="x", parameters=Parameters(binary_data=False)),
                RequestOutput(name="y"),
            ],
            parameters=Parameters(binary_data_output=True),
        )
        default, per_output = resolve_binary_outputs(request)
        assert default is True
        assert per_output == {"x": False, "y": True}


class TestPerOutputEncoding:
    """Mixed binary/JSON outputs in one framed response."""

    def test_mixed_outputs(self):
        response = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="a",
                    datatype="FP32",
                    shape=[2],
                    data=np.array([1.0, 2.0], dtype=np.float32),
                ),
                ResponseOutput(
                    name="b",
                    datatype="FP32",
                    shape=[2],
                    data=np.array([3.0, 4.0], dtype=np.float32),
                ),
            ],
        )
        body, header_length = encode_binary_response(
            response, default_binary=False, per_output={"a": True, "b": False}
        )
        header = orjson.loads(body[:header_length])

        out_a, out_b = header["outputs"]
        assert out_a["parameters"]["binary_data_size"] == 8
        assert "data" not in out_a
        assert out_b["data"] == [3.0, 4.0]
        assert "binary_data_size" not in out_b.get("parameters", {})

        # Only the binary output's bytes follow the header.
        restored = np.frombuffer(body[header_length : header_length + 8], dtype="<f4")
        np.testing.assert_array_equal(restored, [1.0, 2.0])
        assert len(body) == header_length + 8

    def test_list_data_inlined_for_json_outputs(self):
        response = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="b", datatype="FP32", shape=[2], data=TensorData([5.0, 6.0])
                ),
            ],
        )
        body, header_length = encode_binary_response(
            response, default_binary=False, per_output={"b": False}
        )
        header = orjson.loads(body[:header_length])
        assert header["outputs"][0]["data"] == [5.0, 6.0]
        assert len(body) == header_length


class TestBytesFraming:
    """BYTES tensors use length-prefixed element framing (KServe).

    The codec bridge: UTF-8-decodable wire elements parse to the same
    str-element data the JSON path produces (so NumpyCodec decodes them
    identically); non-UTF-8 binary falls back to the codec's
    packed-buffer representation.
    """

    def _bytes_body(self, elements: list[bytes], declared_count: int | None = None):
        import struct

        packed = b"".join(struct.pack("<I", len(e)) + e for e in elements)
        count = declared_count if declared_count is not None else len(elements)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "s",
                        "shape": [count],
                        "datatype": "BYTES",
                        "parameters": {"binary_data_size": len(packed)},
                    }
                ]
            }
        )
        return header + packed, len(header)

    def test_bytes_roundtrip(self):
        elements = [b"hello", b"world"]
        body, header_length = self._bytes_body(elements)
        request, _ = parse_binary_request(body, header_length)

        # UTF-8-decodable elements parse to the JSON-equivalent str form
        data = _unwrap(request.inputs[0].data)
        assert list(data) == ["hello", "world"]

        # ... which NumpyCodec decodes exactly like the JSON input
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.types import RequestInput

        json_equivalent = NumpyCodec.decode_input(
            RequestInput(name="s", datatype="BYTES", shape=[2], data=["hello", "world"])
        )
        np.testing.assert_array_equal(np.asarray(data), json_equivalent.reshape(-1))

        # Response side: BYTES outputs frame each element again.
        response = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="s", datatype="BYTES", shape=[2], data=TensorData(elements)
                ),
            ],
        )
        out_body, out_header_length = encode_binary_response(
            response, default_binary=True
        )
        header = orjson.loads(out_body[:out_header_length])
        assert header["outputs"][0]["datatype"] == "BYTES"
        assert header["outputs"][0]["parameters"]["binary_data_size"] == (4 + 5 + 4 + 5)
        restored = _unpack_region(out_body, out_header_length, 18)
        assert restored == elements

    def test_non_utf8_codec_roundtrip_preserves_bytes(self):
        # The reviewer-prescribed regression: wire -> parse -> NumPy
        # decode -> NumPy encode_output -> binary response must equal
        # the original bytes, including trailing NULs and non-UTF-8
        # values.  Opaque elements take the object-dtype path: exact
        # values and lengths, no fixed-width truncation or NUL
        # stripping anywhere in the chain.
        from mlserver.codecs.numpy import NumpyCodec

        elements = [b"\xff", b"abc", b"\xff\x00"]
        body, header_length = self._bytes_body(elements)
        request, _ = parse_binary_request(body, header_length)
        inp = request.inputs[0]

        # The parser stores opaque bytes as an object-dtype array
        data = _unwrap(inp.data)
        assert isinstance(data, np.ndarray) and data.dtype == object
        assert data.reshape(-1).tolist() == elements

        # NumPy decode keeps every element exact
        decoded = NumpyCodec.decode_input(inp)
        assert decoded.reshape(-1).tolist() == elements

        # NumPy encode_output returns the exact bytes (no surrogate
        # crash, no NUL loss)
        output = NumpyCodec.encode_output("s", decoded)
        assert _unwrap(output.data) == elements

        # The binary response frames them back byte-exactly
        response = InferenceResponse(model_name="m", outputs=[output])
        encoded, n = encode_binary_response(response, default_binary=True)
        assert _unpack_region(encoded, n, len(encoded) - n) == elements

        # A direct echo (no numpy) is also exact
        echo = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name=inp.name,
                    datatype=inp.datatype,
                    shape=inp.shape,
                    data=inp.data,
                )
            ],
        )
        encoded, n = encode_binary_response(echo, default_binary=True)
        assert _unpack_region(encoded, n, len(encoded) - n) == elements

    def test_utf8_trailing_nul_codec_roundtrip(self):
        # Round-6 condition 1: UTF-8 values with trailing NULs keep
        # exact element semantics (the Unicode path would strip them).
        from mlserver.codecs.numpy import NumpyCodec

        elements = [b"a\x00", b"abc"]
        body, header_length = self._bytes_body(elements)
        request, _ = parse_binary_request(body, header_length)
        inp = request.inputs[0]

        decoded = NumpyCodec.decode_input(inp)
        assert decoded.reshape(-1).tolist() == elements

        output = NumpyCodec.encode_output("s", decoded)
        assert getattr(output.data, "root", output.data) == elements

        response = InferenceResponse(model_name="m", outputs=[output])
        encoded, n = encode_binary_response(response, default_binary=True)
        assert _unpack_region(encoded, n, len(encoded) - n) == elements

    def test_logical_elements_survive_batching(self):
        # Round-6 condition 2: the logical-element flag survives the
        # object-array -> list conversion that batching preparation
        # performs, so the codec never reinterprets the elements as
        # legacy packed buffers.
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.grpc.buffers import restore_list_inputs

        elements = [b"\xff", b"abc", b"\xff\x00"]
        body, header_length = self._bytes_body(elements)
        request, _ = parse_binary_request(body, header_length)
        inp = request.inputs[0]

        restore_list_inputs(request)
        data = getattr(inp.data, "root", inp.data)
        assert isinstance(data, list)  # container changed
        assert data == elements

        decoded = NumpyCodec.decode_input(inp)
        assert decoded.size == 3
        assert decoded.reshape(-1).tolist() == elements

    def test_merged_logical_batch_decodes_exactly(self):
        # Adaptive batching aggregates duplicate parameters into lists
        # (True + True -> [True, True]); the logical-element mode must
        # survive aggregation, and the merged batch must decode exactly.
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.batching.requests import BatchedRequests
        from mlserver.grpc.buffers import restore_list_inputs

        def prepared(elements):
            body, header_length = self._bytes_body(elements)
            request, _ = parse_binary_request(body, header_length)
            restore_list_inputs(request)
            return request

        r1 = prepared([b"\xff", b"abc", b"\xff\x00"])
        r2 = prepared([b"\x01\x00", b"xy"])
        merged = BatchedRequests({"a": r1, "b": r2}).merged_request

        decoded = NumpyCodec.decode_input(merged.inputs[0])
        assert decoded.size == 5
        assert decoded.reshape(-1).tolist() == [
            b"\xff",
            b"abc",
            b"\xff\x00",
            b"\x01\x00",
            b"xy",
        ]

    def test_mixed_logical_legacy_batch_normalizes(self):
        # Mixed representations share one codec mode after merge:
        # legacy packed buffers split into elements by the codec's
        # itemsize, so the merged batch decodes coherently.
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.batching.requests import BatchedRequests
        from mlserver.grpc.buffers import restore_list_inputs
        from mlserver.types import RequestInput

        body, header_length = self._bytes_body([b"\xff", b"abc"])
        logical, _ = parse_binary_request(body, header_length)
        restore_list_inputs(logical)

        legacy = InferenceRequest(
            inputs=[
                RequestInput(
                    name="s",
                    datatype="BYTES",
                    shape=[2, 1],
                    data=TensorData([b"\x01\x02"]),
                )
            ]
        )
        merged = BatchedRequests({"a": logical, "b": legacy}).merged_request

        decoded = NumpyCodec.decode_input(merged.inputs[0])
        assert decoded.size == 4
        assert decoded.reshape(-1).tolist() == [b"\xff", b"abc", b"\x01", b"\x02"]

    def test_legacy_only_batch_keeps_packed_interpretation(self):
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.batching.requests import BatchedRequests
        from mlserver.types import RequestInput

        def legacy(buffer):
            return InferenceRequest(
                inputs=[
                    RequestInput(
                        name="s",
                        datatype="BYTES",
                        shape=[2, 1],
                        data=TensorData([buffer]),
                    )
                ]
            )

        merged = BatchedRequests(
            {"a": legacy(b"\x01\x02"), "b": legacy(b"\x03\x04")}
        ).merged_request
        decoded = NumpyCodec.decode_input(merged.inputs[0])
        np.testing.assert_array_equal(
            decoded,
            np.array([[b"\x01"], [b"\x02"], [b"\x03"], [b"\x04"]], dtype=bytes),
        )

    def _logical_prepared(self, elements, shape=None):
        import struct

        from mlserver.grpc.buffers import restore_list_inputs

        raw = b"".join(struct.pack("<I", len(v)) + v for v in elements)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "s",
                        "datatype": "BYTES",
                        "shape": shape or [len(elements), 1],
                        "parameters": {"binary_data_size": len(raw)},
                    }
                ]
            }
        )
        request, _ = parse_binary_request(header + raw, len(header))
        restore_list_inputs(request)
        return request

    def _standalone_decoded(self, request):
        """The values the request's own codec contract decodes standalone."""
        inp = request.inputs[0]
        params = inp.parameters
        content_type = (
            getattr(params, "content_type", None) if params is not None else None
        )
        if content_type == "str":
            return StringCodec.decode_input(inp)
        return NumpyCodec.decode_input(inp).reshape(-1).tolist()

    @pytest.mark.parametrize("order", ["logical-first", "other-first"])
    @pytest.mark.parametrize(
        "other_factory",
        [
            # StringCodec elements: complete UTF-8 words as bytes
            lambda: InferenceRequest(inputs=[StringCodec.encode_input("s", ["hello"])]),
            # NumPy JSON-style strings
            lambda: InferenceRequest(
                inputs=[
                    RequestInput(
                        name="s",
                        datatype="BYTES",
                        shape=[1, 1],
                        data=TensorData(["hello"]),
                    )
                ]
            ),
            # Padded legacy S2 packed buffer: a valid two-element
            # buffer whose first element carries insignificant padding
            lambda: InferenceRequest(
                inputs=[
                    RequestInput(
                        name="s",
                        datatype="BYTES",
                        shape=[1, 2],
                        data=TensorData([b"a\x00bc"]),
                    )
                ]
            ),
        ],
        ids=["string-codec", "json-string", "padded-legacy-s2"],
    )
    def test_mixed_codec_batch_preserves_standalone_values(self, other_factory, order):
        """Each entry keeps its standalone decoded values (and types)
        after merging — in both orders — and split predictions match."""
        from mlserver.batching.requests import BatchedRequests

        other = other_factory()
        # The logical entry mirrors the other's dimensions so the batch
        # is shape-compatible; its first element carries a trailing NUL
        # (flagging it as logical).
        other_shape = other.inputs[0].shape
        element_count = int(np.prod(other_shape))
        logical_elements = [b"x\x00"] + [b"yz"] * (element_count - 1)
        logical = self._logical_prepared(logical_elements, shape=other_shape)

        if order == "logical-first":
            requests = {"a": logical, "b": other}
        else:
            requests = {"a": other, "b": logical}

        expected_a = self._standalone_decoded(requests["a"])
        expected_b = self._standalone_decoded(requests["b"])

        merged = BatchedRequests(requests).merged_request
        # The merged input must not carry a per-request content_type: a
        # "str" selector would route the whole merged batch through the
        # string codec and mis-decode the other requests' normalized
        # bytes elements (a cross-request, order-dependent failure).
        merged_params = merged.inputs[0].parameters
        assert merged_params is None or merged_params.content_type is None
        decoded = NumpyCodec.decode_input(merged.inputs[0]).reshape(-1).tolist()

        assert decoded == expected_a + expected_b
        # the merged shape's element count matches the merged data
        merged_shape = merged.inputs[0].shape
        assert len(decoded) == np.prod(merged_shape)

        # Prediction parity: an echo model's split response returns
        # each request its standalone values.
        from mlserver.types import InferenceResponse, ResponseOutput

        batcher = BatchedRequests(requests)
        echo = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="s",
                    datatype="BYTES",
                    shape=merged_shape,
                    data=TensorData(decoded),
                )
            ],
        )
        split = batcher.split_response(echo)
        assert split["a"].outputs[0].data.root == expected_a
        assert split["b"].outputs[0].data.root == expected_b

    def test_shape_data_mismatch_passes_through(self):
        """A declared-shape/data mismatch passes through as flat bytes
        (matching the JSON path's tolerance) instead of crashing."""
        output = ResponseOutput(
            name="foo", datatype="FP32", shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0]
        )
        raw = _encode_output_bytes(output)
        assert raw is not None
        assert len(raw) == 16  # 4 elements x 4 bytes, flat

        # Through the full encoder: framed with binary_data_size=16
        response = InferenceResponse(model_name="m", outputs=[output])
        body, n = encode_binary_response(response, default_binary=True)
        header = orjson.loads(body[:n])
        assert header["outputs"][0]["parameters"]["binary_data_size"] == 16
        assert len(body) == n + 16

    def test_json_clients_cannot_forge_logical_elements_flag(self):
        """The sentinel is a tuple; JSON parameters produce lists."""
        from mlserver.codecs.numpy import _has_logical_elements

        forged = RequestInput(
            name="s",
            datatype="BYTES",
            shape=[2, 1],
            data=TensorData([b"\x01\x02"]),
            parameters=Parameters(
                **{
                    "_mlserver_logical_bytes_elements": [
                        "mlserver-internal",
                        "logical-bytes-elements",
                    ]
                }
            ),
        )
        assert not _has_logical_elements(forged)

        # ... while the parser's own tuple sentinel is honored
        body, header_length = self._bytes_body([b"\xff", b"abc"])
        request, _ = parse_binary_request(body, header_length)
        assert _has_logical_elements(request.inputs[0])

    def test_object_array_json_output_serializes(self):
        """A JSON-resolved BYTES output carrying an object-dtype array
        (a model echoing the parser's representation) serializes: the
        bytes elements decode instead of raising TypeError, which would
        turn the protocol-legal inline fallback into a 500."""
        response = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="s",
                    datatype="BYTES",
                    shape=[1, 2],
                    data=TensorData(np.array([b"hello", b"\xffworld"], dtype=object)),
                ),
            ],
        )
        body, header_length = encode_binary_response(
            response, default_binary=False, per_output={"s": False}
        )
        header = orjson.loads(body[:header_length])
        out = header["outputs"][0]
        assert "binary_data_size" not in (out.get("parameters") or {})
        # Bytes elements decode with replacement, like the list branch
        assert out["data"] == ["hello", "\ufffdworld"]

    def test_oversized_bytes_element_falls_back_to_json(self):
        """A BYTES element >= 4 GiB overflows the 4-byte length prefix
        (struct.error); the output falls back to inline JSON."""

        class _OversizedBytes(bytes):
            """Reports a 4 GiB length while holding one byte.

            struct.pack("<I", len(elem)) raises the same struct.error
            for this as for a real 4 GiB element, so the fallback branch
            is exercised without allocating gigabytes — CI runners
            segfault on the real thing.
            """

            def __len__(self) -> int:
                return 2**32

        huge = _OversizedBytes(b"x")
        response = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="s", datatype="BYTES", shape=[1, 1], data=TensorData([huge])
                ),
            ],
        )
        # Must not raise; the output falls back to inline JSON
        body, header_length = encode_binary_response(response, default_binary=True)
        header = orjson.loads(body[:header_length])
        # The oversized output has no binary_data_size — it went inline
        assert "binary_data_size" not in header["outputs"][0].get("parameters", {})

    def test_bytes_mode_precheck_is_o1(self):
        """The batching pre-check short-circuits without per-element
        classification when no entry carries the logical-elements flag."""
        from mlserver.batching.requests import _normalize_bytes_entries
        from mlserver.types import RequestInput

        # Large legacy batch: the pre-check must not scan the data
        large_legacy = RequestInput(
            name="s",
            datatype="BYTES",
            shape=[1000, 1],
            data=TensorData([b"\x01"] * 1000),
        )
        entries = [(list(large_legacy.data.root), large_legacy)] * 10
        data, normalized = _normalize_bytes_entries(entries)
        assert not normalized
        assert len(data) == 10

    def test_validation_never_iterates_array_payloads(self, mocker):
        # Structural guard for the pydantic list-coercion regression:
        # array payloads must never reach InferenceRequest.model_validate
        # (pydantic's list[Any] union branch iterates them element by
        # element — 786K iterations for an image tensor).
        from mlserver.types import InferenceRequest

        body, header_length = _rfdet_binary_body()
        spy = mocker.spy(InferenceRequest, "model_validate")
        request, _ = parse_binary_request(body, header_length)
        assert spy.call_count == 1
        validated = spy.call_args.args[0]
        for input_spec in validated["inputs"]:
            assert not isinstance(input_spec.get("data"), np.ndarray)
        # ... and the zero-copy view is still attached post-validation
        data = getattr(request.inputs[0].data, "root", request.inputs[0].data)
        assert isinstance(data, np.ndarray)
        assert data.size == 1 * 3 * 512 * 512

    def test_utf8_bytes_still_decode_like_json(self):
        # The UTF-8 path keeps the JSON-equivalence contract
        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.types import RequestInput

        elements = [b"hello", b"world"]
        body, header_length = self._bytes_body(elements)
        request, _ = parse_binary_request(body, header_length)

        json_equivalent = NumpyCodec.decode_input(
            RequestInput(name="s", datatype="BYTES", shape=[2], data=["hello", "world"])
        )
        np.testing.assert_array_equal(
            NumpyCodec.decode_input(request.inputs[0]), json_equivalent
        )

    def test_truncated_prefix_rejected(self):
        import struct

        # Declared size covers the region, but the second element's
        # payload is cut short inside it (4+5+4 = 13 bytes, no payload).
        packed = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "s",
                        "shape": [2],
                        "datatype": "BYTES",
                        "parameters": {"binary_data_size": len(packed)},
                    }
                ]
            }
        )
        body = header + packed
        with pytest.raises(BinaryTensorError, match="Truncated"):
            parse_binary_request(body, len(header))

    def test_element_count_mismatch_rejected(self):
        elements = [b"a", b"b"]
        body, header_length = self._bytes_body(elements, declared_count=3)
        with pytest.raises(BinaryTensorError, match="expected 3"):
            parse_binary_request(body, header_length)

    def test_unicode_and_empty_elements(self):
        elements = ["héllo".encode("utf-8"), b"", b"\x00\x01"]
        body, header_length = self._bytes_body(elements)
        request, _ = parse_binary_request(body, header_length)
        # All UTF-8-decodable (control bytes are valid UTF-8)
        assert list(_unwrap(request.inputs[0].data)) == [
            "héllo",
            "",
            "\x00\x01",
        ]


class TestBytesCodecBridge:
    """The wire framing bridges to NumpyCodec's BYTES representations."""

    def test_codec_packed_output_one_frame_per_element(self):
        # The existing repository fixture from tests/codecs/test_numpy.py:
        # an 'S1' array becomes shape=[2,1], data=[b"\x01\x02"] (packed).
        import math

        from mlserver.codecs.numpy import NumpyCodec

        values = np.array([[b"\x01"], [b"\x02"]], dtype=bytes)
        output = NumpyCodec.encode_output("s", values)
        response = InferenceResponse(model_name="m", outputs=[output])
        body, n = encode_binary_response(response, default_binary=True)

        header = orjson.loads(body[:n])
        # one 4-byte length prefix + one 1-byte element per frame
        assert header["outputs"][0]["parameters"]["binary_data_size"] == (
            (4 + 1) * math.prod(output.shape)
        )
        elems = _unpack_region(body, n, (4 + 1) * math.prod(output.shape))
        assert len(elems) == math.prod(output.shape)
        assert elems == [b"\x01", b"\x02"]

    def test_str_array_output_frames_elements(self):
        from mlserver.codecs.numpy import NumpyCodec

        values = np.array(["foo", "bar"], dtype=str)
        output = NumpyCodec.encode_output("s", values)
        response = InferenceResponse(model_name="m", outputs=[output])
        body, n = encode_binary_response(response, default_binary=True)

        header = orjson.loads(body[:n])
        size = header["outputs"][0]["parameters"]["binary_data_size"]
        elems = _unpack_region(body, n, size)
        assert elems == [b"foo", b"bar"]

    def test_binary_input_decodes_like_json(self):
        import struct

        from mlserver.codecs.numpy import NumpyCodec
        from mlserver.types import RequestInput

        expected = NumpyCodec.decode_input(
            RequestInput(
                name="s", datatype="BYTES", shape=[2, 1], data=["hello", "world"]
            )
        )
        payload = b"".join(struct.pack("<I", len(v)) + v for v in [b"hello", b"world"])
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "s",
                        "datatype": "BYTES",
                        "shape": [2, 1],
                        "parameters": {"binary_data_size": len(payload)},
                    }
                ]
            }
        )
        request, _ = parse_binary_request(header + payload, len(header))
        actual = NumpyCodec.decode_input(request.inputs[0])
        np.testing.assert_array_equal(actual.astype(str), expected.astype(str))

    def test_codec_roundtrip_through_wire(self):
        # encode_output -> wire -> parse -> decode_input == original array
        from mlserver.codecs.numpy import NumpyCodec

        values = np.array([[b"\x01"], [b"\x02"]], dtype=bytes)
        output = NumpyCodec.encode_output("s", values)

        body, n = encode_binary_response(
            InferenceResponse(model_name="m", outputs=[output]), default_binary=True
        )
        header = orjson.loads(body[:n])
        size = header["outputs"][0]["parameters"]["binary_data_size"]

        request_header = orjson.dumps(
            {
                "inputs": [
                    {
                        "name": "s",
                        "datatype": "BYTES",
                        "shape": output.shape,
                        "parameters": {"binary_data_size": size},
                    }
                ]
            }
        )
        request, _ = parse_binary_request(
            request_header + body[n:], len(request_header)
        )
        decoded = NumpyCodec.decode_input(request.inputs[0])
        np.testing.assert_array_equal(decoded.astype(str), values.astype(str))


def _unpack_region(buf: bytes, offset: int, size: int) -> list[bytes]:
    import struct

    elems = []
    pos = offset
    end = offset + size
    while pos < end:
        (elem_size,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        elems.append(buf[pos : pos + elem_size])
        pos += elem_size
    return elems


class TestHeaderStructureValidation:
    """Malformed metadata fails as BinaryTensorError (4xx), not 500."""

    def test_inputs_null(self):
        header = orjson.dumps({"inputs": None})
        with pytest.raises(BinaryTensorError, match="must be a list"):
            parse_binary_request(header, len(header))

    def test_input_item_scalar(self):
        header = orjson.dumps({"inputs": [42]})
        with pytest.raises(BinaryTensorError, match="must be an object"):
            parse_binary_request(header, len(header))

    def test_parameters_not_object(self):
        header = orjson.dumps(
            {"inputs": [{"name": "x", "parameters": 42, "datatype": "FP32"}]}
        )
        with pytest.raises(BinaryTensorError, match="'parameters' must be an object"):
            parse_binary_request(header, len(header))

    def test_invalid_shape(self):
        header = orjson.dumps(
            {
                "inputs": [
                    {"name": "x", "shape": ["a"], "datatype": "FP32"},
                ]
            }
        )
        with pytest.raises(BinaryTensorError, match="invalid shape"):
            parse_binary_request(header, len(header))

    def test_datatype_not_string(self):
        header = orjson.dumps(
            {"inputs": [{"name": "x", "shape": [1], "datatype": [1, 2]}]}
        )
        with pytest.raises(BinaryTensorError, match="'datatype' must be a string"):
            parse_binary_request(header, len(header))

    def test_missing_name_fails_validation_not_500(self):
        """A well-framed input missing 'name' is a schema error (4xx)."""
        tensor = np.zeros(4, dtype=np.float32)
        header = orjson.dumps(
            {
                "inputs": [
                    {
                        "shape": [4],
                        "datatype": "FP32",
                        "parameters": {"binary_data_size": tensor.nbytes},
                    }
                ]
            }
        )
        body = header + tensor.tobytes()
        with pytest.raises(BinaryTensorError, match="Invalid inference request"):
            parse_binary_request(body, len(header))


class TestNestedBytesOutputs:
    """Nested element form (data matching a multi-dimensional shape).

    A model can return nested lists — the plain-JSON path serializes
    them as-is. The binary encoder must match that structure for
    JSON-inlined outputs and frame nested BYTES identically to their
    flat equivalent.
    """

    def _nested_response(self) -> InferenceResponse:
        return InferenceResponse(
            model_name="repro",
            outputs=[
                ResponseOutput(
                    name="number", datatype="FP32", shape=[1], data=TensorData([1.0])
                ),
                ResponseOutput(
                    name="text",
                    datatype="BYTES",
                    shape=[1, 2],
                    data=TensorData([[b"first", b"second"]]),
                ),
            ],
        )

    def test_mixed_negotiation_inlines_nested_bytes_as_json(self):
        """Numeric binary + nested BYTES JSON: no 500, JSON-path structure."""
        import json

        response = self._nested_response()
        body, header_length = encode_binary_response(
            response, per_output={"number": True, "text": False}
        )
        header = json.loads(body[:header_length])
        assert header["outputs"][1]["data"] == [["first", "second"]]
        assert len(body) - header_length == 4  # only the numeric output's bytes

    def test_nested_bytes_frame_identically_to_flat(self):
        """Fully-binary negotiation: nested and flat produce equal wire bytes."""
        import struct

        response = self._nested_response()
        body, header_length = encode_binary_response(
            response, per_output={"number": True, "text": True}
        )
        header = orjson.loads(body[:header_length])
        size = header["outputs"][1]["parameters"]["binary_data_size"]
        assert size == 4 + 5 + 4 + 6  # two length prefixes + both elements

        frames = body[header_length + 4 :]
        (first_len,) = struct.unpack_from("<I", frames, 0)
        first = frames[4 : 4 + first_len]
        (second_len,) = struct.unpack_from("<I", frames, 4 + first_len)
        second = frames[8 + first_len : 8 + first_len + second_len]
        assert (first, second) == (b"first", b"second")

        flat = InferenceResponse(
            model_name="repro",
            outputs=[
                ResponseOutput(
                    name="number", datatype="FP32", shape=[1], data=TensorData([1.0])
                ),
                ResponseOutput(
                    name="text",
                    datatype="BYTES",
                    shape=[1, 2],
                    data=TensorData([b"first", b"second"]),
                ),
            ],
        )
        flat_body, flat_header_length = encode_binary_response(
            flat, per_output={"number": True, "text": True}
        )
        assert body == flat_body
        assert header_length == flat_header_length

    def test_nested_strings_inline_unchanged(self):
        """Nested str elements (no bytes) inline like the JSON path."""
        response = InferenceResponse(
            model_name="m",
            outputs=[
                ResponseOutput(
                    name="t",
                    datatype="BYTES",
                    shape=[1, 2],
                    data=TensorData([["a", "b"]]),
                ),
            ],
        )
        body, header_length = encode_binary_response(response, default_binary=False)
        header = orjson.loads(body[:header_length])
        assert header["outputs"][0]["data"] == [["a", "b"]]
