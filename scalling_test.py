import os
import time
import csv
import numpy as np
from datetime import datetime
from pymilvus import MilvusClient, DataType

DB_PATH = "milvus_synthetic_scale.db"
COLLECTION_NAME = "scaling_lab"
DIM = 512
SCALES = [1000, 10000, 100000]
OUTPUT_CSV = "scaling_benchmark_results.csv"

def generate_synthetic_vectors(num_vectors: int, dim: int = 512) -> list:
    """Generates L2-normalized synthetic vectors matching CLIP's unit hypersphere."""
    raw = np.random.randn(num_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms
    return normalized.tolist()

def run_scale_experiment():
    client = MilvusClient(DB_PATH)
    CAMERAS = ["gate-01", "gate-02", "perimeter-east", "lobby-main", "warehouse-north"]
    CLASSES = ["animal", "person", "vehicle", "nature", "general"]

    results = []

    for target_scale in SCALES:
        print("\n" + "=" * 65)
        print(f"[*] RUNNING SCALING TEST: {target_scale:,} VECTORS")
        print("=" * 65)

        # 1. Reset collection if it already exists
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)

        # 2. Define schema
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field(field_name="camera", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="image_class", datatype=DataType.VARCHAR, max_length=64)

        # 3. Define index parameters
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "efConstruction": 200}
        )

        # 4. Explicitly CREATE the collection in Milvus
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

        # 5. Ingest synthetic data in fast bulk chunks
        chunk_size = 5000 if target_scale >= 50000 else 1000
        print(f"[*] Ingesting {target_scale:,} synthetic vectors in chunks of {chunk_size}...")
        
        t_ingest_start = time.perf_counter()
        for start_idx in range(0, target_scale, chunk_size):
            count = min(chunk_size, target_scale - start_idx)
            vectors = generate_synthetic_vectors(count, dim=DIM)
            
            rows = []
            for j in range(count):
                row_id = start_idx + j
                rows.append({
                    "id": row_id,
                    "vector": vectors[j],
                    "camera": CAMERAS[row_id % len(CAMERAS)],
                    "image_class": CLASSES[row_id % len(CLASSES)]
                })
            client.insert(collection_name=COLLECTION_NAME, data=rows)

        ingest_time = time.perf_counter() - t_ingest_start
        print(f"[+] Ingestion completed in {ingest_time:.2f} seconds.")

        # Ensure collection is loaded into memory for search
        client.load_collection(COLLECTION_NAME)

        # 6. Measure Database File Footprint
        db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0.0

        # 7. Latency Profiling
        query_vecs = generate_synthetic_vectors(20, dim=DIM)

        # Cold search
        t0 = time.perf_counter()
        client.search(COLLECTION_NAME, data=[query_vecs[0]], limit=5)
        cold_latency = (time.perf_counter() - t0) * 1000

        # Warm search
        warm_latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            client.search(COLLECTION_NAME, data=[query_vecs[0]], limit=5)
            warm_latencies.append((time.perf_counter() - t0) * 1000)
        warm_avg = float(np.mean(warm_latencies))

        # p50 / p95 percentile search (100 runs)
        search_latencies = []
        for i in range(100):
            t0 = time.perf_counter()
            client.search(COLLECTION_NAME, data=[query_vecs[i % len(query_vecs)]], limit=5)
            search_latencies.append((time.perf_counter() - t0) * 1000)

        p50 = float(np.percentile(search_latencies, 50))
        p95 = float(np.percentile(search_latencies, 95))

        # Filtered search overhead
        filtered_times = []
        filter_expr = 'camera == "gate-01" and image_class == "person"'
        for i in range(50):
            t0 = time.perf_counter()
            client.search(COLLECTION_NAME, data=[query_vecs[i % len(query_vecs)]], filter=filter_expr, limit=5)
            filtered_times.append((time.perf_counter() - t0) * 1000)
        filtered_avg = float(np.mean(filtered_times))

        scale_record = {
            "scale": target_scale,
            "ingest_time_sec": round(ingest_time, 2),
            "db_size_mb": round(db_size_mb, 2),
            "cold_latency_ms": round(cold_latency, 2),
            "warm_latency_ms": round(warm_avg, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "filtered_latency_ms": round(filtered_avg, 2),
            "filter_overhead_ms": round(filtered_avg - warm_avg, 2)
        }
        results.append(scale_record)

        print(f"Scale: {target_scale:,} | DB Size: {db_size_mb:.1f} MB | p50: {p50:.2f} ms | p95: {p95:.2f} ms | Filter Overhead: {scale_record['filter_overhead_ms']:+.2f} ms")

    # Save to CSV
    if results:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[+] Full scaling comparison written to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    run_scale_experiment()