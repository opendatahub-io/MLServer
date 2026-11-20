"""Generate two fixed ONNX artifacts and a reproducible correctness scenario."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def generate(directory: Path, cuda: bool, width: int = 2048) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    repo = directory / "models"
    repo.mkdir()
    (repo / "settings.json").write_text(
        json.dumps(
            {
                "load_models_at_startup": False,
                "parallel_workers": 1,
            }
        )
    )
    models = []
    for name, multiplier in (("w1-a", 2), ("w1-b", 3)):
        folder = repo / name
        folder.mkdir()

        def tensor(n: str):
            return helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, width])

        weights = numpy_helper.from_array(
            np.eye(width, dtype=np.float32) * multiplier, "W"
        )
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "W"], ["y"])],
            name,
            [tensor("x")],
            [tensor("y")],
            [weights],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        artifact = model.SerializeToString()
        (folder / "model.onnx").write_bytes(artifact)
        extra = {
            "providers": ["CPUExecutionProvider"],
            "session_options": {"intra_op_num_threads": 1, "inter_op_num_threads": 1},
        }
        if cuda:
            extra.update(
                providers=["CUDAExecutionProvider"],
                provider_options={"device_id": 0},
                session_config_entries={"session.disable_cpu_ep_fallback": "1"},
            )
        settings = {
            "name": name,
            "implementation": "mlserver_onnx.OnnxModel",
            "parameters": {"uri": "model.onnx", "version": "1", "extra": extra},
        }
        (folder / "model-settings.json").write_text(json.dumps(settings))
        values = [float(i % 11 - 5) for i in range(width)]
        models.append(
            dict(
                name=name,
                version="1",
                artifact_sha256=hashlib.sha256(artifact).hexdigest(),
                start_s=0 if name == "w1-a" else 3,
                request={
                    "outputs": [{"name": "y"}],
                    "inputs": [
                        dict(name="x", shape=[1, width], datatype="FP32", data=values)
                    ],
                },
                expected=[
                    dict(
                        name="y",
                        shape=[1, width],
                        datatype="FP32",
                        data=[x * multiplier for x in values],
                    )
                ],
            )
        )
    packages = ["mlserver", "mlserver-onnx", "onnx"]
    packages.extend(importlib.metadata.packages_distributions().get("onnxruntime", []))
    config = dict(
        rate=60,
        duration_s=6,
        warmup_s=1,
        concurrency=8,
        deadline_s=2,
        seed=17,
        models=models,
        initial_load=["w1-a"],
        deployment={
            "workers": 1,
            "execution_options": extra,
            "tensor_width": width,
            "packages": {n: importlib.metadata.version(n) for n in packages},
        },
        actions=[{"at_s": 1, "operation": "load", "model": "w1-b"}],
    )
    (directory / "scenario.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--cuda", action="store_true")
    args = cli.parse_args()
    generate(args.directory, args.cuda)
