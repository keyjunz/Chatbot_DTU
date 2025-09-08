# tests/evaluate_reranked_retrieval.py

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- THIẾT LẬP ĐƯỜNG DẪN ĐỂ IMPORT TỪ THƯ MỤC `src` ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- IMPORT CÁC THÀNH PHẦN CẦN THIẾT ---
from src.chatbot.retrieval_system import RetrievalSystem # Import hệ thống truy xuất tinh gọn
from src.chatbot.config import (
    EVAL_SET_PATH,
    EVAL_RESULTS_DIR
)

class RerankedRetrievalEvaluator:
    """
    Đánh giá hiệu suất của toàn bộ chuỗi truy xuất, BAO GỒM CẢ RE-RANKER.
    """
    def __init__(self):
        """Khởi tạo Evaluator và tải RetrievalSystem."""
        self.retrieval_system = RetrievalSystem()
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    def _perform_queries(self, eval_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Thực hiện truy vấn qua RetrievalSystem và thu thập kết quả."""
        results_data = []
        print(f"\n3. Thực hiện truy vấn qua Retrieval System cho {len(eval_data)} câu hỏi...")

        for item in tqdm(eval_data, desc="Đang đánh giá Pipeline"):
            query = item['query']
            expected_doc_id = item['expected_doc_id']

            # Gọi hệ thống truy xuất để lấy context đã được re-rank
            final_ranked_docs = self.retrieval_system.get_ranked_context(query)
            
            # Lấy danh sách ID từ kết quả
            actual_ids = [doc['id'] for doc in final_ranked_docs]

            rank = 0
            if expected_doc_id in actual_ids:
                rank = actual_ids.index(expected_doc_id) + 1

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
        """In báo cáo ra màn hình."""
        hit_rate = df_results['hit'].mean()
        mrr = df_results['reciprocal_rank'].mean()

        print("\n" + "="*50)
        print("--- KẾT QUẢ ĐÁNH GIÁ (SAU KHI CÓ RE-RANKER) ---")
        print(f"-> Tỷ lệ tìm thấy (Hit Rate): {hit_rate:.2%}")
        print(f"-> Thứ hạng Tương hỗ Trung bình (MRR): {mrr:.4f}")
        print("="*50)

        print("\nBảng kết quả chi tiết:")
        print(df_results[['query', 'expected_doc_id', 'rank']].to_string())

        csv_path = EVAL_RESULTS_DIR / "reranked_retrieval_evaluation_details.csv"
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\nĐã lưu kết quả chi tiết vào file: {csv_path}")

    def _create_visualizations(self, df_results: pd.DataFrame):
        """Tạo và lưu các biểu đồ."""
        print("\n4. Đang tạo các biểu đồ trực quan...")
        
        hit_rate = df_results['hit'].mean()
        mrr = df_results['reciprocal_rank'].mean()

        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Biểu đồ 1: Các chỉ số tổng quan
        plt.figure(figsize=(8, 5))
        metrics = {'Hit Rate': hit_rate, 'MRR': mrr}
        ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="magma")
        ax.bar_label(ax.containers[0], fmt='%.4f')
        plt.title('Tổng quan Chất lượng (Retriever + Re-ranker)', fontsize=16, pad=20)
        plt.ylabel('Giá trị', fontsize=12)
        plt.ylim(0, 1.1)
        summary_chart_path = EVAL_RESULTS_DIR / "reranked_summary_metrics.png"
        plt.savefig(summary_chart_path, bbox_inches='tight')
        print(f"   - Đã lưu biểu đồ tổng quan tại: {summary_chart_path}")
        plt.close()

        # Biểu đồ 2: Phân phối thứ hạng
        plt.figure(figsize=(10, 6))
        # Lọc ra các rank hợp lệ để vẽ biểu đồ
        valid_ranks = df_results[df_results['hit'] == 1]['rank'].astype(int)
        # Xác định thứ tự các cột để hiển thị đầy đủ
        rank_order = range(1, len(df_results.iloc[0]['actual_top_k_ids']) + 1)
        ax = sns.countplot(x=valid_ranks, palette='crest', order=rank_order)
        ax.bar_label(ax.containers[0])
        plt.title('Phân phối Thứ hạng (Sau khi Re-rank)', fontsize=16, pad=20)
        plt.xlabel(f'Thứ hạng', fontsize=12)
        plt.ylabel('Số lượng câu hỏi', fontsize=12)
        rank_dist_path = EVAL_RESULTS_DIR / "reranked_rank_distribution.png"
        plt.savefig(rank_dist_path, bbox_inches='tight')
        print(f"   - Đã lưu biểu đồ phân phối thứ hạng tại: {rank_dist_path}")
        plt.close()

    def run(self):
        """Chạy toàn bộ pipeline đánh giá."""
        try:
            with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        except FileNotFoundError:
            print(f"[LỖI] Không tìm thấy file bộ câu hỏi đánh giá tại: {EVAL_SET_PATH}")
            return

        df_results = self._perform_queries(eval_data)
        self._generate_report(df_results)
        self._create_visualizations(df_results)
        print("\n--- QUÁ TRÌNH ĐÁNH GIÁ RE-RANKED HOÀN TẤT ---")

if __name__ == "__main__":
    evaluator = RerankedRetrievalEvaluator()
    evaluator.run()