import json
import random
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python Path để import config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.chatbot.config import PROCESSED_DATA_DIR

# --- CẤU HÌNH ---
INPUT_FILES = [
    "majors_data_enriched.json",
    "faculty_enriched.json",
    "awards_enriched.json"
]
OUTPUT_TRAIN_PATH = PROCESSED_DATA_DIR / "qa.jsonl"
OUTPUT_EVAL_PATH = PROCESSED_DATA_DIR / "eval.jsonl"
NUM_NEGATIVE_SAMPLES_PER_DOC = 4  # số câu hỏi tiêu cực mỗi tài liệu
EVAL_RATIO = 0.1  # 10% dữ liệu cho eval

# --- TEMPLATE CỐ ĐỊNH ---
SYSTEM_PROMPT = "Bạn là một trợ lý AI hữu ích của trường Đại học Duy Tân. Hãy trả lời câu hỏi chỉ dựa trên thông tin được cung cấp trong phần 'Context'."
NEGATIVE_ANSWER = "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu được cung cấp."

# === Sinh dữ liệu tích cực ===
def generate_positive_examples(doc):
    examples = []
    meta = doc.get('metadata', {})
    source_type = meta.get('source_type', '')

    # --- NGÀNH HỌC ---
    if source_type == 'major':
        ten_nganh = meta.get('ten_nganh', '')
        ma_nganh = meta.get('ma_nganh', '')
        to_hop = [s.split(' ')[0] for s in meta.get('to_hop_mon', []) if s]
        to_hop_str = ", ".join(sorted(list(set(to_hop))))
        
        if ten_nganh and ma_nganh:
            examples.append((f"Ngành {ten_nganh} có mã là gì?", f"Mã ngành của {ten_nganh} là {ma_nganh}."))
            examples.append((f"Mã tuyển sinh của ngành {ten_nganh}?", f"Mã tuyển sinh của ngành {ten_nganh} là {ma_nganh}."))
        
        if ten_nganh and to_hop_str:
            examples.append((f"Để học ngành {ten_nganh} cần thi khối nào?", f"Ngành {ten_nganh} xét tuyển các tổ hợp môn: {to_hop_str}."))
            examples.append((f"Các tổ hợp môn xét tuyển vào ngành {ten_nganh} là gì?", f"Để xét tuyển vào ngành {ten_nganh}, bạn có thể sử dụng các tổ hợp môn sau: {to_hop_str}."))
            examples.append((f"Thông tin tuyển sinh về môn thi của ngành {ten_nganh}?", f"Các tổ hợp môn thi vào ngành {ten_nganh} bao gồm: {to_hop_str}."))

    # --- GIẢNG VIÊN ---
    elif source_type == 'faculty':
        name = meta.get('name', '')
        position = meta.get('position', '').lower()
        faculty = meta.get('faculty', '')
        email = meta.get('email', 'Không có dữ liệu')

        if name and position != 'không có dữ liệu' and position != 'giảng viên':
            examples.append((f"Thầy/cô {name} giữ chức vụ gì?", f"{name} là {position} của khoa {faculty}."))
            examples.append((f"Chức vụ của {name} tại trường là gì?", f"Tại trường, {name} giữ chức vụ {position} thuộc khoa {faculty}."))

        if name and email != 'Không có dữ liệu':
            examples.append((f"Email của {name} là gì?", f"Email của {name} là {email}."))
            examples.append((f"Thông tin liên lạc của {name}?", f"Bạn có thể liên lạc với {name} qua email: {email}."))
            examples.append((f"Địa chỉ email của giảng viên {name}?", f"Địa chỉ email của giảng viên {name} là {email}."))

        if name and faculty:
            examples.append((f"{name} công tác tại khoa nào?", f"{name} hiện đang công tác tại Khoa {faculty}."))

    # --- GIẢI THƯỞNG ---
    elif source_type == 'award':
        title = meta.get('title', '')
        year = meta.get('year')
        
        if title and year:
            examples.append((f"Giải thưởng '{title}' được trao vào năm nào?", f"Giải thưởng '{title}' được trao vào năm {year}."))
            examples.append((f"Trường nhận giải '{title}' khi nào?", f"Trường đã nhận được giải thưởng '{title}' vào năm {year}."))

        if title and "SV VN" in title:
            examples.append((f"Trường có thành tích gì ở kỳ thi Olympic Tin học Sinh viên Việt Nam?", f"Tại kỳ thi Olympic Tin học SV VN, trường đã đạt được thành tích: {title}."))

        if title:
            examples.append((f"Thông tin chi tiết về giải thưởng {title}?", f"Về giải thưởng '{title}': {doc['content']}"))

    return examples

# === Sinh dữ liệu tiêu cực ===
def generate_negative_example(current_doc, all_docs):
    other_doc = random.choice(all_docs)
    while other_doc['id'] == current_doc['id']:
        other_doc = random.choice(all_docs)

    positive_examples_from_other = generate_positive_examples(other_doc)
    if not positive_examples_from_other:
        return None

    question, _ = random.choice(positive_examples_from_other)
    return (question, NEGATIVE_ANSWER)

# === Format dữ liệu chuẩn cho LLM ===
def create_message_format(context, question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n'''\n{context}\n'''\n\nDựa vào Context trên, hãy trả lời câu hỏi sau: {question}"},
            {"role": "assistant", "content": answer}
        ]
    }

# === MAIN ===
def main():
    print("--- Bắt đầu quá trình tạo dữ liệu fine-tune ---")
    
    all_docs = []
    source_mapping = {
        "majors_data_enriched.json": "major",
        "faculty_enriched.json": "faculty",
        "awards_enriched.json": "award"
    }

    print("1. Đang tải tất cả các tài liệu đã được làm giàu...")
    for filename, source_type in source_mapping.items():
        try:
            with open(PROCESSED_DATA_DIR / filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item['metadata']['source_type'] = source_type
                all_docs.extend(data)
        except FileNotFoundError:
            print(f"[Cảnh báo] Không tìm thấy file {filename}, bỏ qua.")
            
    if not all_docs:
        print("[LỖI] Không có dữ liệu để xử lý. Dừng lại.")
        return

    print(f"   -> Đã tải thành công {len(all_docs)} tài liệu.")

    print("2. Đang sinh các cặp câu hỏi-đáp (Q&A)...")
    qa_pairs = []
    for doc in all_docs:
        context = doc['content']
        
        for question, answer in generate_positive_examples(doc):
            qa_pairs.append(create_message_format(context, question, answer))

        for _ in range(NUM_NEGATIVE_SAMPLES_PER_DOC):
            neg = generate_negative_example(doc, all_docs)
            if neg:
                q, a = neg
                qa_pairs.append(create_message_format(context, q, a))
                
    print(f"   -> Đã tạo ra tổng cộng {len(qa_pairs)} mẫu dữ liệu.")

    random.shuffle(qa_pairs)

    split_idx = int(len(qa_pairs) * (1 - EVAL_RATIO))
    train_data = qa_pairs[:split_idx]
    eval_data = qa_pairs[split_idx:]

    print(f"   -> Train set: {len(train_data)} mẫu")
    print(f"   -> Eval set: {len(eval_data)} mẫu")

    with open(OUTPUT_TRAIN_PATH, 'w', encoding='utf-8') as f:
        for pair in train_data:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    with open(OUTPUT_EVAL_PATH, 'w', encoding='utf-8') as f:
        for pair in eval_data:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
    print("\n--- ✅ Hoàn tất! Đã tạo qa.jsonl (train) và eval.jsonl (eval). ---")

if __name__ == "__main__":
    main()
