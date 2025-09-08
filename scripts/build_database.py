# scripts/build_database.py

import chromadb
import json
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python Path để có thể import từ src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sentence_transformers import SentenceTransformer
from src.chatbot.config import (
    PROCESSED_DATA_DIR,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    DEVICE
)

def build_chroma_db():
    """
    Hàm này đọc các file dữ liệu đã xử lý, embedding chúng,
    và import vào ChromaDB.
    Nó sẽ kiểm tra trước để không xây dựng lại nếu database đã tồn tại.
    """
    # Bước 1: Kiểm tra xem DB và collection đã tồn tại chưa
    if CHROMA_PATH.exists():
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            # Nếu lấy được collection mà không báo lỗi, nghĩa là nó đã tồn tại
            if client.get_collection(name=COLLECTION_NAME):
                print(f"✅ Cơ sở dữ liệu ChromaDB và collection '{COLLECTION_NAME}' đã tồn tại. Bỏ qua bước xây dựng.")
                return # Thoát khỏi hàm, không làm gì thêm
        except ValueError:
            # Lỗi này có nghĩa là collection chưa tồn tại, chúng ta sẽ tiếp tục để tạo nó
            print(f"   -> Database tồn tại nhưng collection '{COLLECTION_NAME}' chưa có. Bắt đầu tạo...")
            pass
    
    # Nếu thư mục DB chưa tồn tại, hoặc collection chưa có, bắt đầu xây dựng
    print("--- 🏗️ Bắt đầu xây dựng cơ sở dữ liệu ChromaDB ---")
    
    # Bước 2: Tải mô hình embedding
    print(f"1. Đang tải Embedding Model: '{EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

    # Bước 3: Tải tất cả các tài liệu từ các file JSON đã làm giàu
    all_docs = []
    enriched_files = [
        "majors_data_enriched.json",
        "faculty_enriched.json",
        "awards_enriched.json"
    ]
    print("2. Đang đọc các file dữ liệu đã làm giàu...")
    for filename in enriched_files:
        file_path = PROCESSED_DATA_DIR / filename
        if not file_path.exists():
            print(f"   [Cảnh báo] Không tìm thấy file {file_path}, bỏ qua.")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_docs.extend(data)
            print(f"   -> Đã đọc {len(data)} tài liệu từ {filename}.")
    
    if not all_docs:
        print("[LỖI] Không có tài liệu nào để import. Dừng lại.")
        return

    print(f"   -> Tổng cộng có {len(all_docs)} tài liệu để import.")

    # Bước 4: Chuẩn bị dữ liệu cho ChromaDB
    ids = [doc['id'] for doc in all_docs]
    documents = [doc['content'] for doc in all_docs]
    metadatas = [doc['metadata'] for doc in all_docs]

    # Bước 5: Tạo client, collection, và thêm dữ liệu
    print("3. Đang tạo collection và import dữ liệu vào ChromaDB (có thể mất vài phút)...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # (Tùy chọn) Metadata để chỉ định mô hình embedding đã sử dụng
        metadata={"embedding_model": EMBEDDING_MODEL_NAME}
    )
    
    # Thêm dữ liệu theo từng batch nhỏ để tránh quá tải bộ nhớ trên các môi trường yếu
    batch_size = 64
    for i in range(0, len(ids), batch_size):
        end_index = i + batch_size
        print(f"   -> Đang import batch từ {i} đến {end_index-1}...")
        collection.add(
            ids=ids[i:end_index],
            documents=documents[i:end_index],
            metadatas=metadatas[i:end_index]
        )
    
    print(f"--- ✅ Xây dựng và import dữ liệu vào collection '{COLLECTION_NAME}' hoàn tất! ---")

# Cho phép chạy file này độc lập để test
if __name__ == '__main__':
    build_chroma_db()