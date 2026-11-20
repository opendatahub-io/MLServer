# Benchmarking

These tools benchmark MLServer over gRPC and REST. The k6 examples use the
local test server. The mixed-model runner can also target a deployed replica.

## Previously recorded single-model inference results

|      | Requests/sec | Average (ms) | Slowest (ms) | Fastest (ms) |
| ---- | ------------ | ------------ | ------------ | ------------ |
| gRPC | 2259         | 128.46       | 703.84       | 3.09         |
| REST | 1300         | 226.93       | 304.98       | 2.06         |

## Setup

The single-model inference and fixed 20-model benchmarks use k6.
The mixed-model benchmark uses Python through the repository's Poetry environment.
All three scenarios support REST and gRPC. [Install k6](https://k6.io/docs/getting-started/installation/)

## Data

You can find pre-generated requests under the [`/data`](./data) folder.

### Generate

You can re-generate the test requests by using the
[`generator.py`](./generator.py) script:

```shell
python generator.py
```

## Usage

Run the k6 commands from `benchmarking/`. For the 20-model scenario, run
`./clone-models.sh` before starting the server. Start the k6 test server with

```shell
make start-testserver
```

### Inference benchmark

You can kickstart the HTTP and gRPC inference benchmark by running either:

```shell
make benchmark-rest
make benchmark-grpc
```

These will first load a **Scikit-Learn model trained on the Iris dataset**, and
perform an inference benchmark leveraging both the HTTP and gRPC interfaces
(lasting 60s each).
The model has been set up so as to leverage adaptive batching (i.e. grouping
several requests together on the fly).

At the end of the benchmark, the benchmark scenarios will unload the used model
from the MLServer instance.

### Multi-model k6 scenario

From `benchmarking/`, run `./clone-models.sh` before starting the test server.
It creates ten Iris and ten sum-model copies in the benchmark repository.
Run `make benchmark-mms-rest` or `make benchmark-mms-grpc` against a fresh
server for each protocol. Each scenario loads and invokes all twenty copies
and leaves them loaded. A passing run completes all twenty load/inference pairs
and passes all forty status checks. A missing iteration fails the run.
Both protocols use the same model sequence and their corresponding V2 fixtures.

### Mixed-model measurements

The mixed-model runner supports V2 gRPC (the default) and ordinary V2 JSON
REST. It runs against one MLServer replica so repository operations and inference reach
the same catalog. Start with a disposable repository. The scenario performs
the specified model loads and unloads.

To check the runner before collecting a baseline, run the following from the
repository root. These checks cover output validation, missed arrivals, REST
error reporting, and client deadlines with a real worker over both protocols.
They run on CPU and start their own local servers.

```sh
poetry run python -m pytest tests/benchmarking/check_mixed_model.py
```

With the ONNX runtime installed in the Poetry environment, generate the example
and start the server from the root of the checkout.

```sh
poetry run python -m benchmarking.mixed_model.example /tmp/w1-example
poetry run mlserver start /tmp/w1-example/models
```

Run the scenario in a second terminal.

```sh
poetry run python -m benchmarking.mixed_model.run \
  /tmp/w1-example/scenario.json /tmp/w1-run.sqlite --local-resources
```

For REST, run the same scenario against a fresh server with a separate database.

```sh
poetry run python -m benchmarking.mixed_model.run \
  /tmp/w1-example/scenario.json /tmp/w1-rest.sqlite --protocol rest --local-resources
```

The default addresses are `localhost:8081` for gRPC and `http://localhost:8080`
for REST. Override them with `--address`. The recorded deployment identifies
the protocol. For a comparison keep both models, payloads, offered rate,
worker count, and allocation fixed across runs.

The example warms model A, loads B while A serves, then interleaves version 1
of A and B. Their fixed ONNX weights produce different expected values for the
same input. For the qualified CUDA device-zero profile, generate the example
with `--cuda` in an environment with the locked CUDA dependencies and a visible
GPU. Record the allocation, device identity, driver, server settings, source
revision and lockfile with the results.

The scenario specifies the model names, versions, artifact hashes, V2 inputs and
expected output tensors. Each target has a traffic weight and an optional start
time. The runner checks the response ID, model, version, output names, shapes,
types and values. It uses exact comparison for integer tensors and configurable
absolute and relative tolerances for floating point. Use JSON strings for
UTF-8 `BYTES` tensors on either protocol. When using an existing
deployment, supply the hashes of the artifacts actually deployed. The client
cannot discover them from inference responses.

Set the offered rate (model invocations/second), warmup and measurement durations
(seconds), seed, deadline and maximum concurrent client calls. Management
actions use measurement-relative timestamps and have a separate timeout. Each
action records the expected and observed status (`OK`, a gRPC code name,
or for REST failures a detail such as `HTTP_404`). Use a new database and
the same starting catalog for each comparison. If the runner raises an
exception, the database retains completed observations for diagnosis. The
report marks the run `incomplete`, omits throughput estimates (JSON `null`),
and exits nonzero. A `complete` run finished collection. Its per-model outcomes
show whether the replies were correct. Older databases have no completion
marker. The reporter labels them `unrecorded` when the expected number of
measured arrivals is present and `incomplete` when measured arrivals are missing.

- **Offered and completed traffic.** Every scheduled arrival has a database row,
  including arrivals missed because the client was late or at its concurrency
  limit. Scheduling uses the client's monotonic clock. Slow responses do not
  reduce the offered rate. Measurement starts after the warmup interval ends
  and all warmup requests finish. A request due exactly at the end of a phase
  falls outside that phase. Response latency runs from submission to receipt
  or request error. Windowed throughput counts
  correct responses received within the window, including responses to earlier
  arrivals. Outcomes and latency are grouped by intended arrival time, including
  requests whose responses came back after the window.
- **Client and worker outcomes.** The acceptance lower bound counts replies
  matching the request ID, model, version, and expected tensor structure, even
  when their values differ. Malformed or mismatched replies are recorded as
  `wrong_output` and excluded from the bound. A raw gRPC reply must have one
  buffer per output, including an empty buffer for a zero-element output, and
  omit typed contents. For older databases the bound counts only correct replies,
  since they did not record this distinction. A deadline alone cannot tell
  the client whether execution started or finished. `RESOURCE_EXHAUSTED` is a
  rejected RPC. For REST, HTTP 429 counts as rejected, a client timeout as a
  deadline, and other HTTP failures retain their status (for example, `HTTP_503`).
  Other gRPC failures retain their status code. The existing
  `parallel_request_queue` histogram samples registered response waiters,
  including management messages. It cannot measure execution occupancy.
- **Interference and memory.** The summary includes per-model before, during
  and after windows around each measured management operation. These are
  client-observed management intervals. Registry and pool debug logs identify
  preparation, publication, pending/completed cleanup and recovery separately.
  Use `--local-resources` only when the client shares the server's container or
  cgroup. Samples identify the cgroup memory file and each visible GPU UUID.
  GPU memory is device-wide. Each sample is timed outside the prediction path.
  An unavailable memory reading produces a warning. Traffic and sampling of
  the other resources continue.
- **Optional execution trace.** For ordinary unary gRPC and default JSON REST inference
  with parallel workers, set the following in server `settings.json` and capture server
  stdout using either the default text formatter or MLServer structured JSON logging.

  ```json
  {"logging_settings": {"version": 1, "disable_existing_loggers": false,
    "loggers": {"mlserver.parallel.execution": {"level": "DEBUG", "propagate": true}}}}
  ```

  After worker activity settles, use the following command. It associates the
  runner's unique request IDs with internal worker message IDs. Started,
  returned and raised events describe the worker's model call. A worker exit
  can leave a start without a terminal observation. Client and worker clocks
  are independent. The report also computes mean model-call duration from
  paired timestamps within the same worker. Repeat with tracing disabled and
  enabled using the same artifacts, mix, seed, deadlines, and resource settings
  to measure its cost.

  ```sh
  poetry run python -m benchmarking.mixed_model.report /tmp/w1-run.sqlite \
    --worker-log /tmp/mlserver.log
  ```

Keep the database for comparisons. Its `run`, `attempt`, `lifecycle` and
`resource` tables hold the scenario, per-attempt timings, management timeline
and memory samples. A row in `completed_run` records that collection finished.
A correct response from a different model or version fails
the output check. Wrong outputs, unexpected lifecycle results and targets without
any correct measured reply make the runner exit nonzero. Inspect RPC errors,
deadlines and missed arrivals separately. The example load and latency measurements do not set deployment capacity targets.
