import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "tuyensinh"

# Load model embedding
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

# Kết nối ChromaDB (persistent)
chroma_client = chromadb.PersistentClient(path="data/vector_store/chroma_db")
collection = chroma_client.get_collection(COLLECTION_NAME)

def embed_text(text: str):
    return embedder.encode([text])[0].tolist()

queries = [
        "Ngành An toàn thông tin xét tuyển những tổ hợp môn nào?",
    ]


for q in queries:
    query_emb = embed_text(q)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    print(f"\nCâu hỏi: {q}")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"- {doc}")
        print(f"  (metadata: {meta})")