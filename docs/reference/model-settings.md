# Model Settings

In MLServer, each loaded model can be configured separately. Model settings
include metadata about accepted inputs and options such as adaptive batching.

Set the number of [parallel workers](../user-guide/parallel-inference) in the
server's `settings.json` or through `MLSERVER_PARALLEL_WORKERS`. Setting a worker
count per model is deprecated. You can still set model-level `parallel_workers`
to `0` or less to opt that model out of parallel inference and run it in the main
server process. Positive values and omission use the server setting.

For example, set `"parallel_workers": 0` in the `model-settings.json` of a model
that needs its own `predict_stream()` implementation. This lets that model
stream in the server process while other models use the worker pool.

This configuration will usually be provided through a `model-settings.json`
file which **sits next to the model artifacts**.
However, it's also possible to provide this through environment variables
prefixed with `MLSERVER_MODEL_` (e.g. `MLSERVER_MODEL_IMPLEMENTATION`). Note
that, in the latter case, this environment variables will be shared across all
loaded models (unless they get overriden by a `model-settings.json` file).
Additionally, if no `model-settings.json` file is found, MLServer will also try
to load a _"default"_ model from these environment variables.

## Runtime Implementation Security

MLServer operates in one of two security modes when loading custom runtimes:

**PRODUCTION Mode (Production):**
- Active when a trusted runtimes allowlist file exists in the image
- MLServer validates `implementation` against this allowlist before importing
- Images built via `mlserver build` or `mlserver dockerfile` automatically include
  built-in runtimes (e.g., `mlserver_sklearn.SKLearnModel`,
  `mlserver_xgboost.XGBoostModel`, `mlserver_lightgbm.LightGBMModel`,
  `mlserver_onnx.OnnxModel`) in the allowlist
- Custom runtimes require `--allow-runtime module.ClassName` and matching
  `--runtime-path` during image build
- Directory paths must point to importable Python packages with `__init__.py`
- The dotted `module.ClassName` format keeps declarations explicit

**DEVELOPMENT Mode (Development):**
- Active when no allowlist file exists (e.g., running `mlserver start` directly)
- Custom runtimes are dynamically loaded from model folders at runtime
- Simply place your custom runtime `.py` file next to `model-settings.json`
- MLServer automatically discovers and imports the runtime class
- **Warning:** Only use for local development - not for production

The validation applies regardless of whether `implementation` comes from
`model-settings.json` or `MLSERVER_MODEL_IMPLEMENTATION`.

### Troubleshooting trusted runtime validation

If startup or model loading fails with:

`Model implementation 'module.ClassName' is not included in the trusted runtimes allowlist configuration.`

check the following:

- The value is a dotted import path in `module.ClassName` format.
- The runtime package and module are importable in the serving image.
- For custom runtimes in built images, include each runtime with
  `mlserver build --allow-runtime module.ClassName` and a matching
  `--runtime-path`.
- If both environment variables and `model-settings.json` are present, remember
  `model-settings.json` values take precedence per model.

### Migration note for existing custom runtimes

If you previously relied on dynamically importable runtime paths without an
explicit allowlist entry, update your image build pipeline to pass all served
custom runtimes through `--allow-runtime` and include corresponding
`--runtime-path` values. This keeps runtime loading explicit and prevents
accidental execution of unexpected classes.

### Querying runtime security configuration

You can inspect the current runtime security configuration through the
`/v2/runtimes` REST endpoint or the `RuntimeSecurity` gRPC method. This returns
the security mode (`PRODUCTION` or `DEVELOPMENT`) and, when in `PRODUCTION` mode, the
list of allowed model implementations.

**REST Example:**

```bash
curl http://localhost:8080/v2/runtimes
```

**Response (PRODUCTION mode):**

```json
{
  "mode": "PRODUCTION",
  "allowed_model_implementations": [
    "mlserver_sklearn.SKLearnModel",
    "mlserver_xgboost.XGBoostModel",
    "mlserver_lightgbm.LightGBMModel",
    "mlserver_onnx.OnnxModel",
    "models.MyCustomRuntime"
  ]
}
```

**Response (DEVELOPMENT mode):**

```json
{
  "mode": "DEVELOPMENT"
}
```

## Settings

```{eval-rst}

.. autopydantic_settings:: mlserver.settings.ModelSettings
```

## Extra Model Parameters

```{eval-rst}

.. autopydantic_settings:: mlserver.settings.ModelParameters
```
