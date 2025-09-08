import json
import chromadb
from sentence_transformers import SentenceTransformer

DATA_FILES = [
    "data/processed/majors_data_enriched.json",
    "data/processed/faculty_enriched.json",
    "data/processed/awards_enriched.json",
]
COLLECTION_NAME = "tuyensinh"

# Load model embedding
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

# Kết nối ChromaDB (persistent: lưu DB trên ổ cứng)
chroma_client = chromadb.PersistentClient(path="data/vector_store/chroma_db")
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

def embed_text(text: str):
    return embedder.encode([text])[0].tolist()

def clean_metadata(metadata: dict):
    """Chuyển list -> string, None -> '' để tương thích với ChromaDB"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None or value == "Không có dữ liệu":
            cleaned[key] = ""
        elif isinstance(value, list):
            cleaned[key] = ", ".join(map(str, value))
        else:
            cleaned[key] = value
    return cleaned

# Import dữ liệu
for file_path in DATA_FILES:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for record in data:
        text = record["content"]
        embedding = embed_text(text)
        metadata = clean_metadata(record["metadata"])

        collection.add(
            ids=[record["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    print(f"Imported {file_path}")

print("Dữ liệu đã import xong vào ChromaDB (persisted at data/chroma_db)")
