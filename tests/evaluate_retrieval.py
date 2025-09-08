# tests/evaluate_retrieval.py

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- THIẾT LẬP ĐƯỜNG DẪN ĐỂ IMPORT TỪ THƯ MỤC `src` ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- IMPORT CÁC CẤU HÌNH TỪ FILE CONFIG TRUNG TÂM ---
from src.chatbot.config import (
    EVAL_SET_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EVAL_TOP_K,
    EVAL_RESULTS_DIR,
    DEVICE
)


class RetrievalEvaluator:
    """
    Một lớp để đóng gói toàn bộ logic đánh giá hệ thống truy xuất (retrieval).
    """

    def __init__(self):
        """Khởi tạo Evaluator và tải các tài nguyên cần thiết."""
        print("--- Khởi tạo Retrieval Evaluator ---")
        self.embedder = self._load_embedding_model()
        self.collection = self._connect_to_chromadb()
        # Đảm bảo thư mục lưu kết quả tồn tại
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    def _load_embedding_model(self) -> SentenceTransformer:
        """Tải mô hình embedding."""
        print(f"1. Đang tải Embedding Model: {EMBEDDING_MODEL_NAME} trên {DEVICE}...")
        return SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    def _connect_to_chromadb(self) -> chromadb.Collection:
        """Kết nối tới cơ sở dữ liệu ChromaDB."""
        print(f"2. Đang kết nối tới ChromaDB tại: {CHROMA_PATH}...")
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        return client.get_collection(name=COLLECTION_NAME)

    def _perform_queries(self, eval_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Thực hiện truy vấn cho từng câu hỏi trong bộ đánh giá và thu thập kết quả.
        """
        results_data = []
        print(f"\n3. Thực hiện truy vấn cho {len(eval_data)} câu hỏi (Top K = {EVAL_TOP_K})...")

        for item in tqdm(eval_data, desc="Đang đánh giá"):
            query = item['query']
            expected_doc_id = item['expected_doc_id']

            query_embedding = self.embedder.encode(query).tolist()
            retrieved_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=EVAL_TOP_K
            )
            actual_ids = retrieved_results['ids'][0]

            rank = 0
            if expected_doc_id in actual_ids:
                rank = actual_ids.index(expected_doc_id) + 1  # Thứ hạng bắt đầu từ 1

            results_data.append({
                "query": query,
                "expected_doc_id": expected_doc_id,
                "actual_top_k_ids": actual_ids,
                "rank": "Not Found" if rank == 0 else rank,
                "hit": 1 if rank > 0 else 0,
                "reciprocal_rank": 1 / rank if rank > 0 else 0
            })
        return pd.DataFrame(results_data)

    def _generate_report(self, df_results: pd.DataFrame):
        """
        Tính toán các chỉ số và in báo cáo ra màn hình.
        """
        hit_rate_at_k = df_results['hit'].mean()
        mrr = df_results['reciprocal_rank'].mean()

        print("\n" + "="*50)
        print("--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ---")
        print(f"-> Tỷ lệ tìm thấy trong Top-{EVAL_TOP_K} (Hit Rate@{EVAL_TOP_K}): {hit_rate_at_k:.2%}")
        print(f"-> Thứ hạng Tương hỗ Trung bình (MRR): {mrr:.4f}")
        print("="*50)

        print("\nBảng kết quả chi tiết:")
        print(df_results[['query', 'expected_doc_id', 'rank']].to_string())

        csv_path = EVAL_RESULTS_DIR / "retrieval_evaluation_details.csv"
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nĐã lưu kết quả chi tiết vào file: {csv_path}")

    def _create_visualizations(self, df_results: pd.DataFrame):
        """
        Tạo và lưu các biểu đồ trực quan hóa kết quả.
        """
        print("\n4. Đang tạo các biểu đồ trực quan...")
        
        hit_rate_at_k = df_results['hit'].mean()
        mrr = df_results['reciprocal_rank'].mean()

        # Biểu đồ 1: Các chỉ số tổng quan
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(8, 5))
        metrics = {'Hit Rate': hit_rate_at_k, 'MRR': mrr}
        ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
        ax.bar_label(ax.containers[0], fmt='%.4f') # Thêm nhãn giá trị trên cột
        plt.title(f'Tổng quan Chất lượng Truy xuất (Top K = {EVAL_TOP_K})', fontsize=16, pad=20)
        plt.ylabel('Giá trị', fontsize=12)
        plt.ylim(0, 1.1)
        summary_chart_path = EVAL_RESULTS_DIR / "summary_metrics.png"
        plt.savefig(summary_chart_path, bbox_inches='tight')
        print(f"   - Đã lưu biểu đồ tổng quan tại: {summary_chart_path}")
        plt.close()

        # Biểu đồ 2: Phân phối thứ hạng của các kết quả đúng
        plt.figure(figsize=(10, 6))
        found_ranks = df_results[df_results['hit'] == 1]['rank'].astype(int)
        ax = sns.countplot(x=found_ranks, palette='plasma', order=range(1, EVAL_TOP_K + 1))
        ax.bar_label(ax.containers[0]) # Thêm nhãn số lượng trên cột
        plt.title('Phân phối Thứ hạng của các Kết quả Đúng', fontsize=16, pad=20)
        plt.xlabel(f'Thứ hạng trong Top-{EVAL_TOP_K}', fontsize=12)
        plt.ylabel('Số lượng câu hỏi', fontsize=12)
        rank_dist_path = EVAL_RESULTS_DIR / "rank_distribution.png"
        plt.savefig(rank_dist_path, bbox_inches='tight')
        print(f"   - Đã lưu biểu đồ phân phối thứ hạng tại: {rank_dist_path}")
        plt.close()

    def run(self):
        """
        Hàm chính để chạy toàn bộ pipeline đánh giá.
        """
        try:
            with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        except FileNotFoundError:
            print(f"[LỖI] Không tìm thấy file bộ câu hỏi đánh giá tại: {EVAL_SET_PATH}")
            return

        df_results = self._perform_queries(eval_data)
        self._generate_report(df_results)
        self._create_visualizations(df_results)
        print("\n--- QUÁ TRÌNH ĐÁNH GIÁ HOÀN TẤT ---")


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run()