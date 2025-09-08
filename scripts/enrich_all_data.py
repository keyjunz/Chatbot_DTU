import json
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Chạy script này từ thư mục gốc của dự án
PROCESSED_DATA_DIR = "data/processed"
INPUT_FILES = {
    "majors": "majors.json",
    "faculty": "faculty.json",
    "awards": "awards.json"
}
OUTPUT_FILES = {
    "majors": "majors_data_enriched.json",
    "faculty": "faculty_enriched.json",
    "awards": "awards_enriched.json"
}

# --- CÁC HÀM LÀM GIÀU NỘI DUNG ---

def enrich_major_content(item):
    """Làm giàu nội dung cho một ngành học."""
    meta = item['metadata']
    ten_nganh = meta.get('ten_nganh', '')
    ma_nganh = meta.get('ma_nganh', '')
    to_hop = [s.split(' ')[0] for s in meta.get('to_hop_mon', []) if s]
    to_hop_str = ", ".join(sorted(list(set(to_hop))))
    
    return (
        f"Thông tin tuyển sinh ngành {ten_nganh}, mã ngành {ma_nganh}. "
        f"Để xét tuyển vào ngành này, thí sinh có thể sử dụng các tổ hợp môn: {to_hop_str}. "
        f"Đây là một trong các chương trình đào tạo chính thức của trường."
    )

def enrich_faculty_content(item):
    """Làm giàu nội dung cho một giảng viên."""
    meta = item['metadata']
    degree = meta.get('degree', '')
    name = meta.get('name', '')
    position = meta.get('position', 'Không có dữ liệu').lower()
    faculty = meta.get('faculty', '')
    email = meta.get('email', 'Không có dữ liệu')

    content = f"{degree} {name} là {position} thuộc Khoa {faculty}. "
    if email != "Không có dữ liệu":
        content += f"Thông tin liên hệ qua email là: {email}."
    else:
        content += f"Hiện chưa có thông tin email."
    return content

def enrich_award_content(item):
    """Làm giàu nội dung cho một giải thưởng."""
    meta = item['metadata']
    title = meta.get('title', '')
    year = meta.get('year')
    
    content = f"Thành tích nổi bật: Giải thưởng '{title}'"
    if year:
        content += f" được trao vào năm {year}. "
    else:
        content += ". "
    content += f"Chi tiết: {item['content']}."
    return content

def main():
    """Hàm chính để xử lý tất cả các file."""
    enrichment_functions = {
        "majors": enrich_major_content,
        "faculty": enrich_faculty_content,
        "awards": enrich_award_content
    }

    for key, input_filename in INPUT_FILES.items():
        input_path = os.path.join(PROCESSED_DATA_DIR, input_filename)
        output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILES[key])

        print(f"\n--- Đang xử lý file: {input_path} ---")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[LỖI] Không tìm thấy file: {input_path}. Bỏ qua.")
            continue
            
        enriched_data = []
        for item in data:
            new_item = item.copy()
            new_item['content'] = enrichment_functions[key](item)
            enriched_data.append(new_item)
            
        print(f"Đã làm giàu nội dung cho {len(enriched_data)} mục.")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu file mới vào: {output_path}")

    print("\n--- HOÀN TẤT QUÁ TRÌNH LÀM GIÀU DỮ LIỆU ---")

if __name__ == "__main__":
    main()