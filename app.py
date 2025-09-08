# app.py

import gradio as gr
import sys
from pathlib import Path

# --- THIẾT LẬP ĐƯỜNG DẪN ---
# Thêm thư mục gốc của dự án vào Python Path để có thể import từ src và scripts
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# --- BƯỚC 1: XÂY DỰNG DATABASE (NẾU CẦN THIẾT) ---
# Import và chạy hàm build_chroma_db trước khi làm bất cứ điều gì khác.
# Hàm này sẽ tự kiểm tra và chỉ xây dựng DB nếu nó chưa tồn tại.
from scripts.build_database import build_chroma_db
build_chroma_db()

# --- BƯỚC 2: KHỞI TẠO RAG PIPELINE ---
from src.chatbot.pipeline import RAGPipeline

pipeline = None # Khai báo biến pipeline toàn cục
try:
    print("--- 💡 Đang khởi tạo RAG Pipeline. Quá trình này có thể mất vài phút... ---")
    pipeline = RAGPipeline()
    print("✅ RAG Pipeline đã sẵn sàng để nhận câu hỏi!")
except Exception as e:
    # Nếu không tải được pipeline, ứng dụng sẽ báo lỗi nhưng không bị crash
    print(f"[LỖI NGHIÊM TRỌNG] Không thể khởi tạo RAG Pipeline: {e}")
    # Biến pipeline sẽ vẫn là None


# --- BƯỚC 3: LOGIC XỬ LÝ CHAT ---
def chat_response_function(message, history):
    """
    Hàm này được Gradio gọi mỗi khi người dùng gửi một tin nhắn.
    """
    if pipeline is None:
        # Trả về thông báo lỗi nếu pipeline không khởi tạo được
        return "Xin lỗi, chatbot hiện đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
        
    # Gọi pipeline để lấy kết quả (bao gồm câu trả lời và nguồn)
    result = pipeline.get_answer(message)
    bot_response = result['answer']
    
    # Lấy thông tin nguồn và định dạng nó
    sources = result.get('sources', [])
    if sources:
        bot_response += "\n\n---"
        bot_response += "\n\n**🔍 Nguồn thông tin tham khảo:**"
        for i, source in enumerate(sources):
            source_preview = source.replace('\n', ' ').strip()
            bot_response += f"\n1. *{source_preview[:150]}...*"
            
    return bot_response

# --- BƯỚC 4: TẠO GIAO DIỆN VỚI GRADIO ---
chatbot_interface = gr.ChatInterface(
    fn=chat_response_function,
    title="🎓 Chatbot Tư vấn Tuyển sinh Đại học Duy Tân",
    description="Chào mừng bạn! Hãy hỏi tôi bất kỳ câu hỏi nào liên quan đến thông tin tuyển sinh, ngành học, giảng viên, và các thành tích của trường.",
    examples=[
        ["Ai là trưởng khoa Công nghệ thông tin?"],
        ["Ngành Quản trị khách sạn xét tuyển những tổ hợp môn nào?"],
        ["Trường có thành tích gì ở kỳ thi Olympic Tin học?"],
        ["Học phí của trường là bao nhiêu?"]
    ],
    chatbot=gr.Chatbot(
        height=550,
        label="Cuộc trò chuyện",
        show_copy_button=True,
    ),
    textbox=gr.Textbox(
        placeholder="Nhập câu hỏi của bạn và nhấn Enter",
        container=False,
        scale=7
    ),
    retry_btn="Gửi lại",
    undo_btn="Xóa tin nhắn cuối",
    clear_btn="Bắt đầu cuộc trò chuyện mới",
    theme="soft"
)

# --- BƯỚC 5: CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    chatbot_interface.launch()