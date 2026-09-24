import os
import time
import json
import csv
from datetime import datetime
import numpy as np
from config import Config
from models import HFCLIPModel
from vectordb import Project5VectorStore

BENCHMARK_JSON_PATH = "milvus_benchmark_summary.json"
BENCHMARK_CSV_PATH = "milvus_benchmark_summary.csv"

def benchmark_system():
    print("[*] Setting up benchmark suite...")
    model = HFCLIPModel()
    vdb = Project5VectorStore()

    # Query current collection size
    try:
        current_corpus_size = len(vdb.get_indexed_paths())
    except Exception:
        current_corpus_size = "unknown"

    test_queries = [
        "A group of people surf on a very large wave",
        "A black dog is running after a white dog in the snow",
        "A woman holding a baby and making faces at it",
        "A boy does a skateboard trick off a metal plank",
        "a car nearby a bicycle"
    ]
    query_vectors = [model.encode_text(q) for q in test_queries]

    # 1. Cold vs Warm Search Benchmark
    print("\n[1] Measuring Cold vs. Warm Search Latency...")
    t0 = time.perf_counter()
    vdb.search(query_vectors[0], top_k=5)
    cold_latency = (time.perf_counter() - t0) * 1000

    warm_latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        vdb.search(query_vectors[0], top_k=5)
        warm_latencies.append((time.perf_counter() - t0) * 1000)
    avg_warm_latency = float(np.mean(warm_latencies))

    # 2. Percentile Latency (p50 / p95) across 100 iterations
    print("[2] Profiling query latencies across 100 search calls (p50 / p95)...")
    latencies = []
    for i in range(100):
        vec = query_vectors[i % len(query_vectors)]
        t0 = time.perf_counter()
        vdb.search(vec, top_k=Config.DEFAULT_TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    # 3. Unfiltered vs. Filtered Search Latency
    print("[3] Measuring scalar filter overhead...")
    unfiltered_times = []
    filtered_times = []
    test_filter = 'camera == "gate-01" and image_class == "person"'

    for i in range(50):
        vec = query_vectors[i % len(query_vectors)]
        
        # Unfiltered
        t0 = time.perf_counter()
        vdb.search(vec, scalar_filter=None, top_k=5)
        unfiltered_times.append((time.perf_counter() - t0) * 1000)

        # Filtered
        t0 = time.perf_counter()
        vdb.search(vec, scalar_filter=test_filter, top_k=5)
        filtered_times.append((time.perf_counter() - t0) * 1000)

    avg_unfiltered = float(np.mean(unfiltered_times))
    avg_filtered = float(np.mean(filtered_times))
    filter_overhead = avg_filtered - avg_unfiltered

    # Prepare benchmark summary record
    benchmark_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_size": current_corpus_size,
        "index_type": getattr(Config, "INDEX_TYPE", "FLAT"),
        "metric_type": getattr(Config, "METRIC_TYPE", "COSINE"),
        "cold_latency_ms": round(cold_latency, 2),
        "warm_latency_ms": round(avg_warm_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "unfiltered_latency_ms": round(avg_unfiltered, 2),
        "filtered_latency_ms": round(avg_filtered, 2),
        "filter_overhead_ms": round(filter_overhead, 2)
    }

    # Save to JSON history (Appends new runs to list)
    history = []
    if os.path.exists(BENCHMARK_JSON_PATH):
        try:
            with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
        except Exception:
            history = []
    history.append(benchmark_record)

    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save to CSV history (Appends row for side-by-side comparison)
    file_exists = os.path.exists(BENCHMARK_CSV_PATH)
    fieldnames = list(benchmark_record.keys())
    try:
        with open(BENCHMARK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(benchmark_record)
        print(f"\n[+] Benchmark summary saved to '{BENCHMARK_JSON_PATH}' and '{BENCHMARK_CSV_PATH}'.")
    except PermissionError:
        alt_csv = "milvus_benchmark_summary_latest.csv"
        with open(alt_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(benchmark_record)
        print(f"\n[!] '{BENCHMARK_CSV_PATH}' is open in Excel! Saved backup to '{alt_csv}' instead.")

    # Terminal output
    print("\n" + "=" * 60)
    print("         PROJECT 5: MILVUS EXPERIMENT BENCHMARK")
    print("=" * 60)
    print(f"Corpus Size:                {current_corpus_size} vectors")
    print(f"Index & Metric:             {benchmark_record['index_type']} ({benchmark_record['metric_type']})")
    print("-" * 60)
    print(f"Cold Search Latency:        {cold_latency:.2f} ms")
    print(f"Warm Search Latency (Avg):  {avg_warm_latency:.2f} ms")
    print("-" * 60)
    print(f"p50 Query Latency:          {p50:.2f} ms")
    print(f"p95 Query Latency:          {p95:.2f} ms")
    print("-" * 60)
    print(f"Unfiltered Search (Avg):    {avg_unfiltered:.2f} ms")
    print(f"Filtered Search (Avg):      {avg_filtered:.2f} ms")
    print(f"Filter Overhead:            {filter_overhead:+.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_system()