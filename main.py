# main.py

import sys
from pathlib import Path
import textwrap

# Thêm thư mục gốc của dự án vào Python Path để có thể import từ src
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.chatbot.pipeline import RAGPipeline

def run_interactive_chatbot():
    """
    Hàm chính để khởi tạo và chạy chatbot ở chế độ tương tác.
    """
    print("--- 💡 Đang khởi tạo Chatbot RAG (có thể mất vài phút để tải các mô hình) ---")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] Không thể khởi tạo RAG Pipeline: {e}")
        print("Vui lòng kiểm tra lại cấu hình, đường dẫn và các file dữ liệu.")
        return

    print("\n" + "="*70)
    print("✅ Chatbot đã sẵn sàng! Chào mừng bạn đến với hệ thống tư vấn tuyển sinh.")
    print("   - Gõ câu hỏi của bạn và nhấn Enter.")
    print("   - Gõ 'exit' hoặc 'quit' để kết thúc phiên trò chuyện.")
    print("="*70)

    while True:
        try:
            # Nhận câu hỏi từ người dùng
            question = input("\n[👨‍🎓 BẠN HỎI]: ")

            # Điều kiện thoát
            if question.lower() in ['exit', 'quit']:
                print("\n[🤖 BOT]: Cảm ơn bạn đã sử dụng dịch vụ. Tạm biệt!")
                break
            
            # Bỏ qua nếu người dùng không nhập gì
            if not question.strip():
                continue

            print("\n[🤖 BOT]: ⏳ Đang suy nghĩ...")

            # Gọi pipeline để lấy kết quả
            result = pipeline.get_answer(question)
            answer = result['answer']
            sources = result['sources']

            print("\n[🤖 BOT TRẢ LỜI]:")
            # Sử dụng textwrap để in câu trả lời dài một cách đẹp mắt
            print(textwrap.fill(answer, width=70))

            print("\n   --- Nguồn thông tin đã sử dụng ---")
            if sources:
                for i, source in enumerate(sources):
                    source_preview = source.replace('\n', ' ').strip()
                    print(textwrap.fill(f"    [{i+1}] {source_preview}", width=70, subsequent_indent='        '))
            else:
                print("    (Không có nguồn thông tin cụ thể nào được sử dụng)")
            print("   ---------------------------------")

        except KeyboardInterrupt: # Xử lý khi người dùng nhấn Ctrl+C
            print("\n\n[🤖 BOT]: Đã nhận tín hiệu thoát. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n[LỖI] Đã có lỗi xảy ra trong quá trình xử lý: {e}")


if __name__ == "__main__":
    run_interactive_chatbot()