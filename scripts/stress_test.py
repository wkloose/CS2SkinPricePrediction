#!/usr/bin/env python3
import argparse
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import preprocess_data

_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def build_payloads(data_path: str, limit: int = 10) -> list[dict]:
    df = pd.read_csv(data_path)
    X_train, _, _, _, cols = preprocess_data(df)
    sample_df = X_train.head(limit)
    payloads = []
    for idx in range(len(sample_df)):
        row = sample_df.iloc[idx].tolist()
        payloads.append(
            {
                "dataframe_split": {
                    "columns": list(cols),
                    "data": [row],
                }
            }
        )
    return payloads


def send_request(url: str, payload: dict, timeout: float) -> dict:
    session = get_session()
    start = time.perf_counter()
    try:
        resp = session.post(url, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        upstream = resp.headers.get("X-Upstream-Addr", "unknown")
        if resp.status_code == 200:
            body = resp.json()
            preds = body.get("predictions", [])
            return {
                "ok": True,
                "status": resp.status_code,
                "latency_ms": elapsed_ms,
                "upstream": upstream,
                "prediction": preds[0] if preds else None,
                "error": None,
            }
        return {
            "ok": False,
            "status": resp.status_code,
            "latency_ms": elapsed_ms,
            "upstream": upstream,
            "prediction": None,
            "error": resp.text[:200],
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "ok": False,
            "status": 0,
            "latency_ms": elapsed_ms,
            "upstream": "unknown",
            "prediction": None,
            "error": str(exc),
        }


def run_benchmark(
    url: str,
    total_requests: int,
    concurrency: int,
    warmup: int,
    timeout: float,
    label: str,
    data_path: str,
) -> int:
    payloads = build_payloads(data_path, limit=10)

    for i in range(warmup):
        send_request(url, payloads[i % len(payloads)], timeout)

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, url, payloads[i % len(payloads)], timeout)
            for i in range(total_requests)
        ]
        results = [f.result() for f in futures]
    total_duration = time.perf_counter() - start_all

    latencies = np.array([r["latency_ms"] for r in results], dtype=float)
    success_count = sum(1 for r in results if r["ok"])
    error_count = total_requests - success_count
    error_rate = (error_count / total_requests) * 100.0 if total_requests > 0 else 0.0
    throughput_rps = total_requests / total_duration if total_duration > 0 else 0.0

    upstreams = Counter(r["upstream"] for r in results if r["ok"])
    sample_pred = next((r["prediction"] for r in results if r["prediction"] is not None), None)

    print("=" * 64)
    print(f"STRESS TEST RESULT - {label}")
    print("=" * 64)
    print(f"Target URL       : {url}")
    print(f"Total Requests   : {total_requests}")
    print(f"Concurrency      : {concurrency}")
    print(f"Total Duration   : {total_duration:.3f} s")
    print(f"Success / Errors : {success_count} / {error_count} (Error Rate: {error_rate:.2f}%)")
    print(f"Throughput (RPS) : {throughput_rps:.2f} req/s")
    print("-" * 64)
    print(f"Latency Min      : {np.min(latencies):.2f} ms")
    print(f"Latency Avg      : {np.mean(latencies):.2f} ms")
    print(f"Latency p50      : {np.percentile(latencies, 50):.2f} ms")
    print(f"Latency p95      : {np.percentile(latencies, 95):.2f} ms")
    print(f"Latency p99      : {np.percentile(latencies, 99):.2f} ms")
    print(f"Latency Max      : {np.max(latencies):.2f} ms")
    print("-" * 64)
    print("Replica Distribution (X-Upstream-Addr):")
    for addr in sorted(upstreams):
        cnt = upstreams[addr]
        pct = (cnt / success_count) * 100.0 if success_count > 0 else 0.0
        print(f"  - {addr:<22} : {cnt:4d} ({pct:5.1f}%)")
    if sample_pred is not None:
        print(f"Sample Prediction: {sample_pred:.2f} IDR")
    if error_count > 0:
        first_err = next((r["error"] for r in results if not r["ok"]), "")
        print(f"Sample Error     : {first_err}")
    print("=" * 64)

    return 0 if error_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Concurrent stress test for MLflow serving load balancer")
    parser.add_argument("--url", type=str, default="http://localhost:8080/invocations")
    parser.add_argument("--requests", type=int, default=150)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--label", type=str, default="3-Replicas")
    parser.add_argument("--data-path", type=str, default="data/processed/cs2_market_features.csv")
    args = parser.parse_args()

    exit_code = run_benchmark(
        url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        timeout=args.timeout,
        label=args.label,
        data_path=args.data_path,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()