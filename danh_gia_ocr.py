import os
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
import re
import logging
from paddleocr import PaddleOCR


IMAGE_FOLDER = r"C:\Users\Hi\Downloads\Dataset_SVTR\crop_results" 
LABEL_FILE = r"C:\Users\Hi\Downloads\Dataset_SVTR\crop_results\label.txt" 
# ==========================================

logging.getLogger("ppocr").setLevel(logging.ERROR)

print("Đang load mô hình PaddleOCR (Đã tắt oneDNN để fix lỗi C++)...")

ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)

def clean_text(text):
    """Làm sạch text: Xóa dấu gạch ngang, dấu chấm, khoảng trắng"""
    if text is None: return ""
    text = str(text).upper()
    text = re.sub(r'[\-\.\s]', '', text) 
    return text

def hau_xu_ly(text):
    """Thuật toán Hậu xử lý (Post-processing) sửa lỗi AI dựa trên luật biển số VN"""
    if len(text) < 7: 
        return text # Ngắn quá bỏ qua

    if len(text) >= 9 and text.startswith('1') and text[1].isdigit() and text[2].isdigit():
        text = text[1:]
        
    chars = list(text)
    
    # Xử lý Lỗi 1: Ký tự thứ 3 (Index 2) PHẢI LÀ CHỮ CÁI
    if len(chars) >= 3:
        if chars[2] == '8': chars[2] = 'B'
        elif chars[2] == '6': chars[2] = 'G'
        elif chars[2] == '0': chars[2] = 'D'
        elif chars[2] == '5': chars[2] = 'S'
        elif chars[2] == '2': chars[2] = 'Z'
        
    # Từ ký tự thứ 5 (Index 4) đến cuối PHẢI LÀ SỐ
    for i in range(4, len(chars)):
        if chars[i] == 'B': chars[i] = '8'
        elif chars[i] == 'G': chars[i] = '6'
        elif chars[i] == 'S': chars[i] = '5'
        elif chars[i] == 'Z': chars[i] = '2'
        elif chars[i] == 'D' or chars[i] == 'O': chars[i] = '0'
        
    return "".join(chars)

# 3. ĐỌC FILE DỮ LIỆU
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

total_images = 0
correct_predictions = 0

print("\nBẮT ĐẦU ĐÁNH GIÁ 486 ẢNH...\n" + "-"*40)

for line in lines:
    line = line.strip()
    if not line: continue
    
    parts = line.split('\t') 
    if len(parts) < 2: continue
    
    filename = parts[0].strip()
    true_label = parts[1].strip()
    
    img_path = os.path.join(IMAGE_FOLDER, filename)
    
    # Bỏ qua nếu file ảnh không tồn tại
    if not os.path.exists(img_path):
        continue
        
    total_images += 1
    
    result = ocr.ocr(img_path)
    
    predicted_text = ""
    if result and result[0]:
        for res in result[0]:
            predicted_text += res[1][0]
            
    # 5. SO SÁNH
    clean_true = clean_text(true_label)
    
    # Kích hoạt Vũ khí Hậu xử lý
    clean_pred_raw = clean_text(predicted_text) 
    clean_pred = hau_xu_ly(clean_pred_raw) # Đưa qua hàm sửa lỗi
    
    if clean_pred == clean_true:
        correct_predictions += 1
        if clean_pred != clean_pred_raw:
             print(f"[TỰ SỬA ĐÚNG] File: {filename} | AI gốc: {clean_pred_raw} -> Đã sửa: {clean_pred}")
    else:
        print(f"[SAI]  File: {filename} | Nhãn thật: {clean_true} | AI đọc: {clean_pred}")

# ==========================================
# 6. IN BÁO CÁO TỔNG KẾT
# ==========================================
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print("\n" + "="*50)
print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ SVTR (PRE-TRAINED)")
print(f"Tổng số ảnh đã test  : {total_images}")
print(f"Số ảnh đọc ĐÚNG      : {correct_predictions}")
print(f"Số ảnh đọc SAI       : {total_images - correct_predictions}")
print(f"ĐỘ CHÍNH XÁC (ACC)   : {accuracy:.2f}%")
print("="*50)