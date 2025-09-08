from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options # Thêm import này
from bs4 import BeautifulSoup
import pandas as pd
import time

# ----- Cải tiến: Thêm Options cho Chrome để tăng tính ổn định -----
chrome_options = Options()
# chrome_options.add_argument("--headless") # Bỏ comment dòng này nếu bạn muốn chạy ẩn, không hiện trình duyệt
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
# Tùy chọn để tránh bị phát hiện là bot
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)
# Giảm bớt log không cần thiết trên console
chrome_options.add_argument('--log-level=3') 

# Khởi tạo webdriver với các tùy chọn đã thiết lập
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

url = "https://duytan.edu.vn/tuyen-sinh"
driver.get(url)

# ----- XỬ LÝ POP-UP -----
try:
    close_button_main = WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "div.vex-close"))
    )
    close_button_main.click()
    print("✅ Đã đóng pop-up quảng cáo chính.")
    time.sleep(1)
except Exception as e:
    print("ℹ️ Không tìm thấy pop-up quảng cáo chính hoặc nó không xuất hiện.")

# ----- ĐIỀU HƯỚNG TỚI TRANG NGÀNH NGHỀ -----
try:
    print("Đang tìm và click vào mục 'NGÀNH NGHỀ ĐÀO TẠO'...")
    nganh_nghe_link = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "NGÀNH NGHỀ ĐÀO TẠO"))
    )
    nganh_nghe_link.click()
    print("✅ Đã click thành công.")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".nganhnghe-wp"))
    )
    print("✅ Trang danh sách ngành nghề đã tải xong.")
except Exception as e:
    print(f"❌ Lỗi: Không thể click vào mục 'NGÀNH NGHỀ ĐÀO TẠO'. {e}")
    driver.quit()
    exit()

soup = BeautifulSoup(driver.page_source, "html.parser")
data = []

# ------------------ B1: Crawl danh sách ngành ------------------
for block in soup.select(".nganhnghe-wp"):
    truong = block.select_one(".name-box .name-title").get_text(strip=True)
    for accordion in block.select(".accordion__item"):
        bac_dao_tao = accordion.select_one(".nganhnghe-list-header").get_text(strip=True)
        for li in accordion.select("ul li a"):
            nganh = li.get_text(strip=True)
            link = li.get("href")
            if link and link.startswith("/"):
                link = "https://duytan.edu.vn" + link
            
            if not link:
                continue

            # ------------------ B2: Crawl chi tiết ngành ------------------
            try:
                driver.get(link)
                # Tăng thời gian chờ lên 20 giây để xử lý các trang tải chậm
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.box_news_detail"))
                )
                sub_soup = BeautifulSoup(driver.page_source, "html.parser")
                box = sub_soup.select_one("div.box_news_detail")
                
                # Logic lấy dữ liệu vẫn giữ nguyên vì nó đã đủ linh hoạt
                # để xử lý các trang không có bảng hoặc không có chuyên ngành
                if not box:
                    print(f"❌ Không tìm thấy box_news_detail: {link}")
                    continue

                ma_nganh = ""
                to_hop = ""
                table = box.select_one("table")
                if table:
                    data_row = table.select_one("tr:nth-of-type(3)")
                    if data_row:
                        cols = data_row.select("td")
                        if len(cols) >= 3:
                            ma_nganh_tag = cols[1].select_one("strong")
                            if ma_nganh_tag:
                                ma_nganh = ma_nganh_tag.get_text(strip=True)
                            else:
                                ma_nganh = cols[1].get_text(strip=True).split("(")[0]
                            to_hop = cols[2].get_text(" ", strip=True)

                desc_blocks = box.find_all(['p', 'div'], string=lambda t: t and "Mã chuyên ngành:" in t)
                
                if not desc_blocks:
                    data.append([
                        truong, bac_dao_tao, nganh, ma_nganh,
                        "", "", to_hop, "", link
                    ])
                    continue

                for blk in desc_blocks:
                    text = blk.get_text(" ", strip=True)
                    ten_cn = text.split("(")[0].replace("*", "").strip()
                    try:
                        ma_cn = text.split("Mã chuyên ngành:")[-1].replace(")", "").strip()
                    except:
                        ma_cn = ""
                    
                    mo_ta_tag = blk.find_next_sibling("p")
                    mo_ta = mo_ta_tag.get_text(" ", strip=True) if mo_ta_tag else ""
                    
                    link_detail = ""
                    if mo_ta_tag:
                        a_tag = mo_ta_tag.find("a")
                        if a_tag and a_tag.has_attr("href"):
                            href = a_tag["href"]
                            if not href.startswith("http"):
                                link_detail = f"https://duytan.edu.vn/tuyen-sinh/Page/{href}"
                            else:
                                link_detail = href
                    
                    data.append([
                        truong, bac_dao_tao, nganh, ma_nganh,
                        ten_cn, ma_cn, to_hop, mo_ta, link_detail
                    ])

            except Exception as e:
                # Lỗi này giờ chủ yếu sẽ là TimeoutException nếu trang tải quá lâu
                print(f"⚠️ Lỗi khi crawl chi tiết hoặc timeout: {link}, {e}")
                continue

driver.quit()

# ------------------ B3: Xuất dữ liệu ------------------
df = pd.DataFrame(data, columns=[
    "Trường", "Bậc đào tạo", "Ngành", "Mã ngành",
    "Tên chuyên ngành", "Mã chuyên ngành", "Tổ hợp môn",
    "Mô tả", "Link chi tiết"
])
df.drop_duplicates(inplace=True)
df.to_excel("tat_ca_chuyen_nganh_final.xlsx", index=False, engine="openpyxl")
print(" Đã crawl xong, lưu file tat_ca_chuyen_nganh_final.xlsx")