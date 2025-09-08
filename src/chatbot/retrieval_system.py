# src/chatbot/retrieval_system.py

import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào Python Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from typing import List

# Import các cấu hình từ file config trung tâm
from src.chatbot.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    DEVICE,
    N_RETRIEVE_RESULTS,
    N_FINAL_RESULTS
)

class RetrievalSystem:
    """
    Một phiên bản tinh gọn của pipeline, chỉ tập trung vào
    Retriever và Re-ranker, được tối ưu cho việc đánh giá.
    """
    def __init__(self):
        """Khởi tạo và tải các mô hình cần thiết cho việc truy xuất."""
        print("--- Đang khởi tạo Retrieval System (tinh gọn) ---")
        self.embedder = self._load_embedding_model()
        self.collection = self._connect_to_chromadb()
        self.reranker = self._load_reranker_model()
        print("✅ Retrieval System đã sẵn sàng!")

    def _load_embedding_model(self) -> SentenceTransformer:
        """Tải mô hình Bi-Encoder để tạo vector embedding."""
        print(f"1. Đang tải Embedding Model: '{EMBEDDING_MODEL_NAME}' trên '{DEVICE}'...")
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    def _connect_to_chromadb(self) -> chromadb.Collection:
        """Kết nối tới cơ sở dữ liệu vector ChromaDB."""
        print(f"2. Đang kết nối tới ChromaDB tại: '{CHROMA_PATH}'...")
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        return client.get_collection(name=COLLECTION_NAME)

    def _load_reranker_model(self) -> CrossEncoder:
        """Tải mô hình Cross-Encoder để tái xếp hạng."""
        print(f"3. Đang tải Re-ranker Model: '{RERANKER_MODEL_NAME}' trên '{DEVICE}'...")
        return CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=DEVICE)

    def get_ranked_context(self, query: str) -> List[dict]:
        """
        Thực hiện truy xuất và tái xếp hạng, sau đó trả về
        thông tin đầy đủ (id, content, metadata) của các tài liệu cuối cùng.
        """
        # Bước 1: Truy xuất ban đầu từ ChromaDB
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RETRIEVE_RESULTS,
            include=["metadatas", "documents"]
        )

        # Tạo một danh sách các dictionary chứa thông tin tài liệu ban đầu
        initial_docs = []
        for i in range(len(results['ids'][0])):
            initial_docs.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })

        if not initial_docs:
            return []

        # Bước 2: Tái xếp hạng các tài liệu đã truy xuất
        context_contents = [doc['content'] for doc in initial_docs]
        pairs = [[query, content] for content in context_contents]
        scores = self.reranker.predict(pairs)
        
        # Sắp xếp lại danh sách tài liệu ban đầu dựa trên điểm số mới
        reranked_docs = sorted(zip(scores, initial_docs), reverse=True)
        
        # Trả về N_FINAL_RESULTS tài liệu tốt nhất
        return [doc for score, doc in reranked_docs][:N_FINAL_RESULTS]