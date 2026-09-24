import os
import hashlib
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType
from config import Config

class Project5VectorStore:
    def __init__(self, db_path: str = Config.DB_PATH, collection_name: str = Config.COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = MilvusClient(db_path)
        self._ensure_collection_and_index()

    def _ensure_collection_and_index(self):
        """Creates the collection with an explicit scalar schema and HNSW index."""
        if not self.client.has_collection(self.collection_name):
            print(f"[*] Initializing collection '{self.collection_name}' with scalar schema...")
            
            # 1. Define explicit schema
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
            schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="image_class", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="camera", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="capture_date", datatype=DataType.VARCHAR, max_length=32)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)

            # 2. Configure HNSW index parameters
            index_params = self.client.prepare_index_params()
            if Config.INDEX_TYPE == "FLAT":
                index_params.add_index(
                    field_name="vector",
                    metric_type=Config.METRIC_TYPE,
                    index_type="FLAT",
                    params={}
                )
            else:  # HNSW default
                index_params.add_index(
                    field_name="vector",
                    metric_type=Config.METRIC_TYPE,
                    index_type="HNSW",
                    params={"M": 16, "efConstruction": 200}
                )

            # 3. Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"[+] Collection '{self.collection_name}' and HNSW index created successfully.")
        
        self.client.load_collection(self.collection_name)

    @staticmethod
    def generate_id_from_path(file_path: str) -> int:
        return int(hashlib.md5(os.path.abspath(file_path).encode("utf-8")).hexdigest()[:8], 16)

    def get_indexed_paths(self) -> set:
        """Retrieves existing file paths for idempotent, resume-safe ingestion."""
        try:
            results = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=["file_path"],
                limit=16384
            )
            return {os.path.abspath(item["file_path"]) for item in results}
        except Exception:
            return set()

    def insert_batch(self, records: List[Dict[str, Any]]):
        """Inserts vectors alongside scalar metadata."""
        self.client.insert(collection_name=self.collection_name, data=records)
        self.client.load_collection(self.collection_name)

    def search(
        self, 
        query_vector: List[float], 
        scalar_filter: Optional[str] = None, 
        top_k: int = Config.DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Executes ANN search with optional scalar boolean filtering."""
        output_fields = ["file_path", "image_class", "camera", "capture_date", "source"]
        
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=scalar_filter if scalar_filter else "",
            limit=top_k,
            output_fields=output_fields
        )

        if not search_res or len(search_res[0]) == 0:
            return []

        formatted = []
        for hit in search_res[0]:
            formatted.append({
                "id": hit["id"],
                "score": float(hit["distance"]),
                "file_path": hit["entity"].get("file_path"),
                "image_class": hit["entity"].get("image_class"),
                "camera": hit["entity"].get("camera"),
                "capture_date": hit["entity"].get("capture_date"),
                "source": hit["entity"].get("source")
            })
        return formatted