"""Summarize a mixed-model run and correlate optional worker traces."""

import argparse
import json
import math
import sqlite3
from pathlib import Path


def summarize_window(
    db: sqlite3.Connection,
    model: str,
    version: str,
    start: float,
    end: float,
    *,
    report_throughput: bool = True,
) -> dict:
    selector = "FROM attempt WHERE phase='measure' AND model=? AND version=? "
    args = (model, version)
    cohort = selector + "AND due>=? AND due<? "
    params = (*args, start, end)
    counts = dict(
        db.execute("SELECT outcome,COUNT(*) " + cohort + "GROUP BY outcome", params)
    )
    elapsed = end - start
    arrivals = sum(counts.values())
    correct_in_window = db.execute(
        "SELECT COUNT(*) "
        + selector
        + "AND outcome='correct' AND finished>=? AND finished<?",
        params,
    ).fetchone()[0]
    timings = cohort + "AND finished IS NOT NULL "
    # Older runs did not distinguish malformed replies from wrong tensor values.
    columns = {col[1] for col in db.execute("PRAGMA table_info(attempt)")}
    evidence = (
        "verified_reply" if "verified_reply" in columns else "(outcome='correct')"
    )
    total, accepted = db.execute(
        f"SELECT COUNT(*),COALESCE(SUM({evidence}),0) " + timings, params
    ).fetchone()
    latency = {}
    for pct in (50, 95, 99):
        if total:
            latency[f"p{pct}"] = db.execute(
                "SELECT 1000*(finished-submitted) "
                + timings
                + "ORDER BY 1 LIMIT 1 OFFSET ?",
                (*params, math.ceil(pct * total / 100) - 1),
            ).fetchone()[0]
    return dict(
        offered=arrivals,
        submitted=total,
        accepted_lower_bound_from_replies=accepted,
        outcomes=counts,
        client_latency_ms=latency,
        correct_responses_in_window=correct_in_window,
        correct_responses_per_s=(
            correct_in_window / elapsed if elapsed and report_throughput else None
        ),
    )


def read_worker_log(db: sqlite3.Connection, path: Path, run_id: str) -> list[dict]:
    # Keep trace correlation off the serving path and on disk for long runs.
    db.execute("DROP TABLE IF EXISTS temp.worker_event")
    db.execute(
        "CREATE TEMP TABLE worker_event(id TEXT, message TEXT, pid INTEGER, "
        "event TEXT, ns INTEGER, PRIMARY KEY(message,pid,event))"
    )
    with path.open() as stream:
        for line in stream:
            if "worker_execution " not in line:
                continue
            if line.startswith("{"):
                line = json.loads(line).get("message", "")
            _, marker, payload = line.partition("worker_execution ")
            if marker:
                event = json.loads(payload)
                if str(event["request"]).startswith(run_id + "."):
                    db.execute(
                        "INSERT OR IGNORE INTO worker_event VALUES(?,?,?,?,?)",
                        (
                            event["request"],
                            event["message"],
                            event["pid"],
                            event["event"],
                            event["ns"],
                        ),
                    )
    db.execute("CREATE INDEX temp.worker_id ON worker_event(id)")
    return [
        dict(
            model=model,
            version=version,
            client_outcome=outcome,
            worker_event=event,
            observations=n,
        )
        for model, version, outcome, event, n in db.execute(
            "SELECT a.model,a.version,a.outcome,e.event,COUNT(*) FROM worker_event e "
            "JOIN attempt a ON a.id=e.id WHERE a.phase='measure' GROUP BY 1,2,3,4"
        )
    ]


def summary(db: sqlite3.Connection, worker_log: Path | None = None) -> dict:
    row = db.execute("SELECT id,address,scenario FROM run").fetchone()
    if row is None:
        raise ValueError("No benchmark run was recorded")
    run_id, address, config = row
    config = json.loads(config)
    duration = config["duration_s"]
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='completed_run'").fetchone():
        completed = db.execute(
            "SELECT 1 FROM completed_run WHERE id=?", (run_id,)
        ).fetchone()
        run_status = "complete" if completed else "incomplete"
    else:
        # Older databases have no completion marker. Missing scheduled arrivals
        # prove an interruption; a full set cannot prove the run finished.
        arrivals = db.execute(
            "SELECT COUNT(*) FROM attempt WHERE phase='measure' AND due<?", (duration,)
        ).fetchone()[0]
        run_status = (
            "incomplete" if arrivals / config["rate"] < duration else "unrecorded"
        )
    windows = [("all", 0, duration)]
    actions = []
    for name, op, due, start, finish, status, expected in db.execute(
        "SELECT model,operation,due,submitted,finished,status,expected FROM lifecycle "
        "WHERE phase='measure' ORDER BY submitted"
    ):
        actions.append(
            dict(
                model=name,
                operation=op,
                due_s=due,
                start_s=start,
                end_s=finish,
                status=status,
                expected=expected,
            )
        )
        label = f"{len(actions)}:{op}:{name}"
        for part, left, right in (
            ("before", 0, start),
            ("during", start, finish),
            ("after", finish, duration),
        ):
            if left < min(right, duration):
                windows.append((f"{label}:{part}", left, min(right, duration)))
    result = dict(
        run=run_id,
        status=run_status,
        address=address,
        deployment=config["deployment"],
        actions=actions,
        models=[
            dict(
                model=m["name"],
                version=m["version"],
                windows={
                    name: summarize_window(
                        db,
                        m["name"],
                        m["version"],
                        a,
                        b,
                        report_throughput=run_status != "incomplete",
                    )
                    for name, a, b in windows
                },
            )
            for m in config["models"]
        ],
        memory=[
            dict(source=s, subject=i, min_bytes=lo, max_bytes=hi, samples=n)
            for s, i, lo, hi, n in db.execute(
                "SELECT source,subject,MIN(bytes),MAX(bytes),COUNT(*) "
                "FROM resource GROUP BY 1,2"
            )
        ],
    )
    if worker_log:
        result["observed_worker_events"] = read_worker_log(db, worker_log, run_id)
        result["worker_calls"] = [
            dict(
                model=name,
                version=version,
                started=started,
                observed_terminal=terminal,
                mean_worker_call_ms=mean_ms,
            )
            for name, version, started, terminal, mean_ms in db.execute(
                "SELECT a.model,a.version,COUNT(*),COUNT(t.ns),AVG(t.ns-s.ns)/1e6 "
                "FROM worker_event s JOIN attempt a ON a.id=s.id "
                "LEFT JOIN worker_event t ON t.message=s.message AND t.pid=s.pid "
                "AND t.event IN ('returned','raised') "
                "WHERE s.event='started' AND a.phase='measure' GROUP BY 1,2"
            )
        ]
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("database", type=Path)
    cli.add_argument("--worker-log", type=Path)
    args = cli.parse_args()
    with sqlite3.connect(
        f"{args.database.resolve().as_uri()}?mode=ro", uri=True
    ) as connection:
        result = summary(connection, args.worker_log)
        print(json.dumps(result, indent=2))
        if result["status"] == "incomplete":
            raise SystemExit(1)
