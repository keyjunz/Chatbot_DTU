# src/chatbot/pipeline.py

import sys
from pathlib import Path

# Thêm thư mục gốc của dự án vào Python Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from typing import List

# Import RetrievalSystem đã được tách riêng
from src.chatbot.retrieval_system import RetrievalSystem
# Import các cấu hình cần thiết
from src.chatbot.config import LLM_MODEL_NAME, DEVICE, LORA_ADAPTER_PATH

class RAGPipeline:
    """
    Class đóng gói toàn bộ pipeline RAG, sử dụng RetrievalSystem và LLM.
    Phiên bản này có khả năng tải LoRA adapter đã được fine-tune.
    """
    def __init__(self):
        """
        Khởi tạo pipeline bằng cách tải RetrievalSystem và mô hình LLM.
        """
        print("--- Đang khởi tạo RAG Pipeline (Đầy đủ) ---")
        self.retrieval_system = RetrievalSystem()
        self.llm_pipe = self._load_llm()
        print("RAG Pipeline đã sẵn sàng!")

    def _load_llm(self) -> pipeline:
        """
        Tải mô hình ngôn ngữ lớn (LLM).
        Nếu tìm thấy LoRA adapter đã được fine-tune, nó sẽ được áp dụng.
        Ngược lại, sẽ sử dụng model gốc.
        """
        print(f"4. Đang tải LLM và Tokenizer...")

        # --- BƯỚC 1: TẢI MODEL GỐC VỚI QUANTIZATION 4-BIT ---
        # Cấu hình quantization phải giống hệt như lúc train
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        
        print(f"   -> Đang tải model gốc: '{LLM_MODEL_NAME}' trên '{DEVICE}'...")
        base_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map=DEVICE,
            trust_remote_code=True
        )

        # --- BƯỚC 2: KIỂM TRA VÀ ÁP DỤNG LoRA ADAPTER ---
        if LORA_ADAPTER_PATH and LORA_ADAPTER_PATH.exists():
            print(f"   -> Tìm thấy LoRA adapter! Đang áp dụng từ: '{LORA_ADAPTER_PATH}'")
            # Tải adapter và áp dụng lên model gốc
            model = PeftModel.from_pretrained(base_model, str(LORA_ADAPTER_PATH))
            # Hợp nhất các trọng số của adapter vào model gốc để tăng tốc độ inference.
            # Sau bước này, model sẽ hoạt động như một model đầy đủ đã được fine-tune.
            print("   -> Đang hợp nhất (merging) LoRA adapter...")
            model = model.merge_and_unload()
            print("   -> Hợp nhất thành công!")
        else:
            print("   -> Không tìm thấy LoRA adapter. Sử dụng model gốc.")
            model = base_model
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    def _build_prompt(self, query: str, context_contents: List[str]) -> str:
        """Xây dựng Prompt để gửi cho LLM."""
        context_str = "\n\n---\n\n".join(context_contents)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý AI hữu ích của trường Đại học Duy Tân. "
                    "Nhiệm vụ của bạn là trả lời câu hỏi của sinh viên và phụ huynh một cách chính xác "
                    "dựa trên thông tin được cung cấp trong phần 'Context'. "
                    "Hãy trả lời một cách ngắn gọn, đi thẳng vào vấn đề. "
                    "Nếu thông tin không có trong Context, hãy trả lời: "
                    "'Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu được cung cấp.'"
                )
            },
            {
                "role": "user",
                "content": f"""
                Context:
                '''
                {context_str}
                '''

                Dựa vào Context trên, hãy trả lời câu hỏi sau: {query}
                """
            }
        ]
        return self.llm_pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def get_answer(self, query: str) -> dict:
        """
        Hàm chính để nhận câu hỏi và trả về câu trả lời cuối cùng từ LLM.
        """
        # Bước 1 & 2: Lấy context đã được truy xuất và tái xếp hạng
        final_ranked_docs = self.retrieval_system.get_ranked_context(query)
        
        if not final_ranked_docs:
            return {
                "answer": "Xin lỗi, tôi không tìm thấy bất kỳ thông tin nào liên quan đến câu hỏi của bạn.",
                "sources": []
            }

        final_context_contents = [doc['content'] for doc in final_ranked_docs]
        
        # Bước 3: Xây dựng prompt
        prompt = self._build_prompt(query, final_context_contents)
        
        # Bước 4: Sinh câu trả lời từ LLM
        generation_args = {
            "max_new_tokens": 512,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": True,
        }
        output = self.llm_pipe(prompt, **generation_args)
        answer = output[0]['generated_text'].strip()
        
        return {
            "answer": answer,
            "sources": final_context_contents
        }