import os
import re
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from config import Config
from models import HFCLIPModel
from vectordb import Project5VectorStore

CAMERAS = ["gate-01", "gate-02", "perimeter-east", "lobby-main", "warehouse-north"]
SOURCES = ["cctv_archive", "field_survey", "mobile_upload", "dock_sensor"]
DATES = ["2026-01-15", "2026-02-20", "2026-03-10", "2026-04-05", "2026-05-12"]

def derive_class_from_caption(caption: str) -> str:
    caption_lower = caption.lower()
    if re.search(r"\b(dog|dogs|cat|puppy|animal|bird)\b", caption_lower):
        return "animal"
    elif re.search(r"\b(boy|girl|man|woman|person|people|child|children|climber)\b", caption_lower):
        return "person"
    elif re.search(r"\b(car|bike|bicycle|truck|skateboard|motorcycle)\b", caption_lower):
        return "vehicle"
    elif re.search(r"\b(water|beach|ocean|mountain|snow|grass|forest|field)\b", caption_lower):
        return "nature"
    return "general"

def run_ingestion():
    model = HFCLIPModel()
    vdb = Project5VectorStore()

    indexed_paths = vdb.get_indexed_paths()
    print(f"[*] Found {len(indexed_paths)} records already present in Milvus.")

    print(f"[*] Initializing stream for full Flickr8k corpus...")
    streamed_ds = load_dataset("jxie/flickr8k", split="train", streaming=True)

    batch_pil_images = []
    batch_metadata = []
    
    # Process sequentially without loading entire 8K into memory at once
    total_expected = 6000
    pbar = tqdm(total=total_expected, desc="Ingesting Full 6K")

    for idx, item in enumerate(streamed_ds):
        pbar.update(1)
        fname = f"flickr_{idx:04d}.jpg"
        local_path = os.path.abspath(os.path.join(Config.LOCAL_IMAGE_DIR, fname))

        # Save image locally if missing
        if not os.path.exists(local_path):
            try:
                item["image"].convert("RGB").save(local_path, "JPEG")
            except Exception as e:
                print(f"[!] Error saving {local_path}: {e}")
                continue

        # Skip if already in database
        if local_path in indexed_paths:
            continue

        raw_caption = item.get("caption_0") or item.get("caption") or ""
        caption_str = raw_caption[0] if isinstance(raw_caption, list) and raw_caption else str(raw_caption)

        image_class = derive_class_from_caption(caption_str)
        camera = CAMERAS[idx % len(CAMERAS)]
        capture_date = DATES[idx % len(DATES)]
        source = SOURCES[idx % len(SOURCES)]

        try:
            img = Image.open(local_path).convert("RGB")
            batch_pil_images.append(img)
            batch_metadata.append({
                "id": vdb.generate_id_from_path(local_path),
                "file_path": local_path,
                "image_class": image_class,
                "camera": camera,
                "capture_date": capture_date,
                "source": source
            })
        except Exception as e:
            print(f"[!] Skipping corrupted file {local_path}: {e}")
            continue

        # Commit whenever batch size is reached
        if len(batch_pil_images) >= Config.BATCH_SIZE:
            vectors = model.encode_images(batch_pil_images)
            milvus_rows = []
            for meta, vec in zip(batch_metadata, vectors):
                row = dict(meta)
                row["vector"] = vec
                milvus_rows.append(row)
            
            vdb.insert_batch(milvus_rows)
            batch_pil_images.clear()
            batch_metadata.clear()

    # Flush remaining records
    if batch_pil_images:
        vectors = model.encode_images(batch_pil_images)
        milvus_rows = []
        for meta, vec in zip(batch_metadata, vectors):
            row = dict(meta)
            row["vector"] = vec
            milvus_rows.append(row)
        vdb.insert_batch(milvus_rows)

    pbar.close()
    print("[+] Complete 6K ingestion and indexing finished successfully.")

if __name__ == "__main__":
    run_ingestion()