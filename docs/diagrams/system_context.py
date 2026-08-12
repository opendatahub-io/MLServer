"""MLServer System Context Diagram (C4-style)."""

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.compute import Server
from diagrams.onprem.monitoring import Prometheus
from diagrams.onprem.queue import Kafka
from diagrams.programming.framework import FastAPI
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.8",
    "nodesep": "1.0",
    "ranksep": "1.0",
    "splines": "ortho",
}

with Diagram(
    "MLServer — System Context",
    filename="/Users/imran/Projects/MLServer/docs/diagrams/system_context",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    outformat="png",
):
    # External actors
    client = Users("Client\nApplications")
    kafka_ext = Kafka("Kafka\nBroker")
    prometheus = Prometheus("Prometheus")
    model_store = Storage("Model Artifact\nStore")

    with Cluster(
        "MLServer Boundary", graph_attr={"bgcolor": "#E8F4FD", "pencolor": "#4A90D9"}
    ):

        with Cluster("Transport Layer"):
            rest = FastAPI("REST API\n:8080")
            grpc = Server("gRPC API\n:8081")
            metrics = FastAPI("Metrics\n:8082")
            kafka_srv = Server("Kafka\nConsumer")

        with Cluster("Core Engine"):
            dataplane = Rack("DataPlane\n(inference orchestration)")
            middleware = Rack("Middleware\n(hooks, caching)")

        with Cluster("Model Management"):
            registry = SQL("MultiModel\nRegistry")
            batcher = Rack("Adaptive\nBatcher")

        with Cluster("Execution"):
            pool = Rack("Inference\nPool")
            workers = Server("Worker\nProcesses")

    # Client paths
    client >> Edge(label="HTTP/1.1 + JSON", color="#4A90D9", style="bold") >> rest
    client >> Edge(label="HTTP/2 + Protobuf", color="#2C5F8A", style="bold") >> grpc

    # Transport → Core
    rest >> Edge(color="#4A90D9") >> dataplane
    grpc >> Edge(color="#2C5F8A") >> dataplane
    kafka_srv >> Edge(color="#9B59B6") >> dataplane

    # Core → Model Management
    dataplane >> Edge(color="#7ED321") >> middleware
    middleware >> Edge(color="#7ED321") >> registry
    registry >> Edge(color="#7ED321") >> batcher

    # Model Management → Execution
    batcher >> Edge(color="#F5A623") >> pool
    pool >> Edge(label="dispatch", color="#F5A623") >> workers

    # External integrations
    (
        kafka_ext
        >> Edge(label="consume / produce", color="#9B59B6", style="dashed")
        >> kafka_srv
    )
    (
        prometheus
        >> Edge(label="scrape /metrics", color="#D0021B", style="dashed")
        >> metrics
    )

    # Model loading
    (
        workers
        >> Edge(label="load artifacts", color="#8B572A", style="dashed")
        >> model_store
    )
    (
        registry
        >> Edge(label="discover models", color="#8B572A", style="dashed")
        >> model_store
    )
