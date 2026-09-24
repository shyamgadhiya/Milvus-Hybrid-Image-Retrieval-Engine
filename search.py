import os
from PIL import Image
from config import Config
from models import HFCLIPModel
from vectordb import Project5VectorStore

def calibrate_score(raw_cosine: float) -> float:
    """
    Transforms raw CLIP cosine similarities into an intuitive human percentage.
    CLIP cosine scores between image and text generally span ~0.15 (unrelated) to ~0.35+ (strong match).
    """
    min_thresh = 0.15
    max_thresh = 0.35
    clipped = max(min_thresh, min(raw_cosine, max_thresh))
    pct = ((clipped - min_thresh) / (max_thresh - min_thresh)) * 100.0
    return pct

def run_interactive_search(display_images: bool = True):
    model = HFCLIPModel()
    vdb = Project5VectorStore()

    print("\n" + "=" * 65)
    print("   PROJECT 5: HYBRID FILTERED IMAGE RETRIEVAL ENGINE")
    print("=" * 65)
    print("Available Cameras: gate-01, gate-02, perimeter-east, lobby-main, warehouse-north")
    print("Available Classes: animal, person, vehicle, nature, general")
    print("Filter syntax:     camera == 'gate-01' and image_class == 'animal'")
    print("=" * 65 + "\n")

    while True:
        try:
            query_text = input("Query > ").strip()
            if query_text.lower() in ("exit", "quit", "q"):
                break
            if not query_text:
                continue

            filter_expr = input("Filter (press enter for none) > ").strip()
            filter_expr = filter_expr if filter_expr else None

            # 1. Encode text query
            query_vec = model.encode_text(query_text)

            # 2. Hybrid search against Milvus
            hits = vdb.search(query_vec, scalar_filter=filter_expr, top_k=Config.DEFAULT_TOP_K)

            print(f"\n--- Results for '{query_text}' [Filter: {filter_expr or 'None'}] ---")
            if not hits:
                print("No images matched the criteria.\n")
                continue

            # 3. Print metadata and calibrated match percentage
            for rank, res in enumerate(hits, start=1):
                file_name = os.path.basename(res["file_path"])
                calibrated = calibrate_score(res["score"])
                
                print(f"Top {rank}: {file_name:<20} | Raw Cosine: {res['score']:.4f} | Match: {calibrated:>5.1f}%")
                print(f"       Class: {res['image_class']:<10} | Cam: {res['camera']:<15} | Date: {res['capture_date']}")
                print(f"       Path: {res['file_path']}")

                # 4. Open images using your previous display logic
                if display_images and os.path.exists(res["file_path"]):
                    try:
                        Image.open(res["file_path"]).show(title=f"Rank {rank}: {file_name}")
                    except Exception as e:
                        print(f"       [!] Could not display image: {e}")

            print("=" * 65 + "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_interactive_search(display_images=True)