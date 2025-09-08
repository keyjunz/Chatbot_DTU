# src/chatbot/config.py

import torch
from pathlib import Path

# --- PATHS & DIRECTORIES ---
# Sử dụng pathlib để đảm bảo các đường dẫn hoạt động trên mọi hệ điều hành
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
CHROMA_PATH = VECTOR_STORE_DIR / "chroma_db"

TESTS_DIR = ROOT_DIR / "tests"
EVAL_RESULTS_DIR = TESTS_DIR / "evaluation_results"
EVAL_SET_PATH = TESTS_DIR / "evaluation_set.json" # [THAY ĐỔI] Sửa lại đường dẫn cho đúng cấu trúc

# --- MODEL IDENTIFIERS ---
# Tên các mô hình được tải từ Hugging Face
EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" # [LƯU Ý] Model này lớn, có thể chạy chậm trên CPU miễn phí của Spaces.
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # [THAY ĐỔI] Sử dụng model gốc mà bạn đã fine-tune

# --- FINE-TUNED MODEL PATH ---
# [THAY ĐỔI] Đây là phần quan trọng nhất để deploy.
# Nó sẽ trỏ đến ID của adapter trên Hugging Face Hub, không phải đường dẫn cục bộ.
# Vui lòng thay 'your-hf-username' bằng tên đăng nhập Hugging Face của bạn.
YOUR_HF_USERNAME = "Keyjun" 
ADAPTER_NAME = "Chatbot-deloy" # Tên repo adapter bạn đã tải lên
LORA_ADAPTER_PATH = f"{YOUR_HF_USERNAME}/{ADAPTER_NAME}"

# --- CHROMA DATABASE SETTINGS ---
COLLECTION_NAME = "tuyensinh"

# --- RAG PIPELINE PARAMETERS ---
N_RETRIEVE_RESULTS = 10
N_FINAL_RESULTS = 3

# --- EVALUATION PARAMETERS ---
EVAL_TOP_K = 5

# --- COMPUTATIONAL DEVICE ---
# Tự động xác định sử dụng GPU ('cuda') nếu có, ngược lại sử dụng CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SERVER DEPLOYMENT SETTINGS ---
# Các biến này hiện không được sử dụng trực tiếp nếu bạn dùng Gradio
# nhưng giữ lại cũng không sao.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860 # [THAY ĐỔI] Cổng mặc định của Gradio trên Spaces là 7860