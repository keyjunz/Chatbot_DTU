# upload_adapter.py
from huggingface_hub import HfApi

YOUR_HF_USERNAME = "Keyjun" 
LOCAL_ADAPTER_PATH = "src/chatbot/models/lora_adapter" # Đường dẫn đến thư mục adapter đã train
# ============================================

api = HfApi()
adapter_name = LOCAL_ADAPTER_PATH.split('/')[-1]
repo_id = f"{YOUR_HF_USERNAME}/{adapter_name}"

print(f"1. Đang tạo repository '{repo_id}' trên Hugging Face Hub...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print(f"2. Đang tải các file từ '{LOCAL_ADAPTER_PATH}' lên...")
api.upload_folder(
    folder_path=LOCAL_ADAPTER_PATH,
    repo_id=repo_id,
    repo_type="model"
)
print(f"Tải lên thành công! Truy cập adapter của bạn tại: https://huggingface.co/{repo_id}")