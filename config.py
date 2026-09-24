import os
import torch

class Config:
    # Model Configuration
    MODEL_NAME = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM = 512
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Milvus Configuration
    DB_PATH = "milvus_project5_lab.db"
    COLLECTION_NAME = "image_catalog"
    METRIC_TYPE = "COSINE"
    INDEX_TYPE = "FLAT"
    
    # Storage Configuration
    LOCAL_IMAGE_DIR = "./image_corpus"
    BATCH_SIZE = 32
    DEFAULT_TOP_K = 3

os.makedirs(Config.LOCAL_IMAGE_DIR, exist_ok=True)