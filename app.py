# app.py

import gradio as gr
import sys
from pathlib import Path

# --- THIẾT LẬP ĐƯỜNG DẪN ---
# Thêm thư mục gốc của dự án vào Python Path để có thể import từ src
# Điều này rất quan trọng để ứng dụng chạy được trên Hugging Face Spaces
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.chatbot.pipeline import RAGPipeline

# --- KHỞI TẠO PIPELINE (CHẠY MỘT LẦN DUY NHẤT) ---
# Bọc trong try-except để xử lý các lỗi có thể xảy ra khi tải mô hình,
# ví dụ như hết bộ nhớ hoặc lỗi mạng.
try:
    print("--- Đang khởi tạo RAG Pipeline. Quá trình này có thể mất vài phút... ---")
    pipeline = RAGPipeline()
    print("RAG Pipeline đã sẵn sàng để nhận câu hỏi!")
except Exception as e:
    # Nếu không tải được pipeline, ứng dụng sẽ báo lỗi nhưng không bị crash
    print(f"[LỖI NGHIÊM TRỌNG] Không thể khởi tạo RAG Pipeline: {e}")
    pipeline = None

# --- LOGIC XỬ LÝ CHAT ---
def chat_response_function(message, history):
    """
    Hàm này được Gradio gọi mỗi khi người dùng gửi một tin nhắn.
    Nó nhận tin nhắn mới và lịch sử chat, sau đó trả về câu trả lời của bot.
    
    Args:
        message (str): Tin nhắn mới của người dùng.
        history (List[List[str]]): Lịch sử cuộc trò chuyện.

    Returns:
        str: Câu trả lời của bot.
    """
    if pipeline is None:
        # Trả về thông báo lỗi nếu pipeline không khởi tạo được
        return "Xin lỗi, chatbot hiện đang gặp sự cố kỹ thuật. Vui lòng thử lại sau."
        
    # Gọi pipeline để lấy kết quả (bao gồm câu trả lời và nguồn)
    result = pipeline.get_answer(message)
    bot_response = result['answer']
    
    # Lấy thông tin nguồn và định dạng nó một cách đẹp mắt
    sources = result.get('sources', [])
    if sources:
        # Thêm tiêu đề cho phần nguồn
        bot_response += "\n\n---"
        bot_response += "\n\n**🔍 Nguồn thông tin tham khảo:**"
        for i, source in enumerate(sources):
            # Trích một đoạn ngắn của nguồn để hiển thị
            source_preview = source.replace('\n', ' ').strip()
            # Sử dụng Markdown để định dạng danh sách
            bot_response += f"\n1. *{source_preview[:150]}...*"
            
    return bot_response

# --- TẠO GIAO DIỆN VỚI GRADIO ---
# Sử dụng gr.ChatInterface, một cách nhanh chóng để tạo một giao diện chat hoàn chỉnh
# Nó tự động quản lý lịch sử, ô nhập liệu, nút gửi, v.v.
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
    theme="soft" # Giao diện mềm mại, dễ nhìn
)

# --- CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    # iface.launch() sẽ tạo một web server trên máy cục bộ của bạn
    # Khi deploy lên Spaces, nó sẽ tự động chạy trên cổng 7860
    chatbot_interface.launch()