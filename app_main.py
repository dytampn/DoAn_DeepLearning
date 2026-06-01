import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import logging
from ultralytics import RTDETR
from paddleocr import PaddleOCR

# ==========================================
# 1. KHỞI TẠO CÁC MÔ HÌNH AI
# ==========================================
print("Đang khởi động hệ thống AI...")

# Mô hình 1: RT-DETR (Nhận diện vị trí)
print("- Đang tải RT-DETR...")
det_model = RTDETR('best.pt') 

# Mô hình 2: SVTR - PaddleOCR (Đọc chữ)
logging.getLogger("ppocr").setLevel(logging.ERROR)
print("- Đang tải SVTR...")
ocr_model = PaddleOCR(use_textline_orientation=False, lang='en', use_mkldnn=False)

print("Hệ thống sẵn sàng!\n")

# ==========================================
# 2. HÀM HẬU XỬ LÝ (SỬA LỖI OCR & ĐỊNH DẠNG BIỂN SỐ)
# ==========================================
def hau_xu_ly(text):
    if not text or len(text) < 7: return text 
    text = str(text).upper()
    text = re.sub(r'[\-\.\s]', '', text) 
        
    if len(text) >= 9 and text.startswith('1') and text[1].isdigit() and text[2].isdigit():
        text = text[1:]
        
    chars = list(text)
    if len(chars) >= 3:
        if chars[2] == '8': chars[2] = 'B'
        elif chars[2] == '6': chars[2] = 'G'
        elif chars[2] == '0': chars[2] = 'D'
        elif chars[2] == '5': chars[2] = 'S'
        elif chars[2] == '2': chars[2] = 'Z'
        
    for i in range(4, len(chars)):
        if chars[i] == 'B': chars[i] = '8'
        elif chars[i] == 'G': chars[i] = '6'
        elif chars[i] == 'S': chars[i] = '5'
        elif chars[i] == 'Z': chars[i] = '2'
        elif chars[i] in ['D', 'O', 'Q']: chars[i] = '0'
        
    return "".join(chars)

def format_bien_so(text):
    if not text:
        return text
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # 1. Xe máy 5 số: 2 số tỉnh + 1 chữ + 1 số + 5 số (VD: 59S745708 -> 59-S7-457.08)
    match = re.match(r'^(\d{2})([A-Z])(\d)(\d{5})$', clean)
    if match:
        return f"{match.group(1)}-{match.group(2)}{match.group(3)}-{match.group(4)[:3]}.{match.group(4)[3:]}"
        
    # 2. Xe máy 4 số có quận: 2 số tỉnh + 1 chữ + 1 số + 4 số (VD: 59S74570 -> 59-S7-45.70)
    match = re.match(r'^(\d{2})([A-Z])(\d)(\d{4})$', clean)
    if match:
        return f"{match.group(1)}-{match.group(2)}{match.group(3)}-{match.group(4)[:2]}.{match.group(4)[2:]}"

    # 3. Ô tô 5 số (1 chữ): 2 số tỉnh + 1 chữ + 5 số (VD: 51A12345 -> 51A-123.45)
    match = re.match(r'^(\d{2})([A-Z])(\d{5})$', clean)
    if match:
        return f"{match.group(1)}{match.group(2)}-{match.group(3)[:3]}.{match.group(3)[3:]}"

    # 4. Ô tô 4 số hoặc biển 4 số cũ: 2 số tỉnh + 1 chữ + 4 số (VD: 59Z7812 -> 59-Z-7812)
    match = re.match(r'^(\d{2})([A-Z])(\d{4})$', clean)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # 5. Ô tô 5 số hệ chữ mới/nước ngoài: 2 số tỉnh + 2 chữ + 5 số (VD: 51LD12345 -> 51-LD-123.45)
    match = re.match(r'^(\d{2})([A-Z]{2})(\d{5})$', clean)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)[:3]}.{match.group(3)[3:]}"

    # 6. Ô tô 4 số hệ chữ mới/nước ngoài: 2 số tỉnh + 2 chữ + 4 số (VD: 51LD1234 -> 51-LD-12.34)
    match = re.match(r'^(\d{2})([A-Z]{2})(\d{4})$', clean)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)[:2]}.{match.group(3)[2:]}"

    # Định dạng mặc định nếu không khớp mẫu trên
    if len(clean) >= 3 and clean[:2].isdigit():
        return f"{clean[:2]}-{clean[2:]}"
    return text

# ==========================================
# 3. HÀM TIỀN XỬ LÝ & NHẬN DIỆN BIỂN SỐ
# ==========================================
def get_iou(boxA, boxB):
    """Tính toán Intersection over Union (IoU) giữa hai bounding box"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def preprocess_plate(crop_img):
    """Tiền xử lý ảnh biển số: Phóng to, Tăng tương phản (CLAHE) & Làm nét để sửa lỗi mờ"""
    if crop_img is None or crop_img.size == 0:
        return crop_img
        
    h, w = crop_img.shape[:2]
    if w < 180:
        scale = 180.0 / w
        crop_img = cv2.resize(crop_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
    yuv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    crop_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 3. Lọc sắc nét bằng bộ lọc ma trận (Sharpening Filter)
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    crop_img = cv2.filter2D(crop_img, -1, kernel)
    
    return crop_img

def run_ocr_on_crop(crop_img):
    """Thực hiện tiền xử lý ảnh cắt biển số và chạy mô hình OCR"""
    processed = preprocess_plate(crop_img)
    ocr_result = ocr_model.ocr(processed)
    
    text_raw = ""
    if ocr_result and ocr_result[0]:
        for res in ocr_result[0]:
            text_raw += res[1][0]
            
    final_text = hau_xu_ly(text_raw)
    return format_bien_so(final_text)

def process_single_image(img):
    """Xử lý đơn lẻ cho file Ảnh gốc (luôn chạy OCR trên mọi biển phát hiện được)"""
    results = det_model(img, conf=0.4, verbose=False)
    annotated_img = img.copy()
    
    boxes = results[0].boxes.xyxy.cpu().numpy() 
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop_img = img[y1:y2, x1:x2]
        
        # Chạy OCR đã được tăng cường chất lượng ảnh
        formatted_text = run_ocr_on_crop(crop_img)
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated_img, formatted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5) 
        cv2.putText(annotated_img, formatted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        print(f"-> Đã nhận diện (Ảnh): {formatted_text}")
        
    return annotated_img

def process_frame_with_tracking(img, frame_count, tracked_plates):
    """Hàm xử lý video/webcam: Chạy RT-DETR trên mọi frame, nhưng chạy OCR cách khung hình để chống delay"""
    # 1. Phát hiện biển số bằng RT-DETR
    results = det_model(img, conf=0.4, verbose=False)
    annotated_img = img.copy()
    
    current_boxes = results[0].boxes.xyxy.cpu().numpy()
    new_tracked_plates = []
    
    for cur_box in current_boxes:
        cur_box = list(map(int, cur_box))
        x1, y1, x2, y2 = cur_box
        
        # Tìm xem box này có khớp với biển số nào đang được track từ frame trước không
        best_match = None
        best_iou = 0.25 # Ngưỡng IoU trùng lặp để xem là cùng 1 biển số
        
        for plate in tracked_plates:
            iou = get_iou(cur_box, plate['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = plate
                
        if best_match is not None:
            # Nếu trùng, giữ nguyên chữ đã nhận diện trước đó
            text = best_match['text']
            
            # Cứ mỗi 10 frame, hoặc nếu chữ đang rỗng thì chạy OCR lại để cập nhật/làm rõ chữ
            if frame_count % 10 == 0 or not text:
                crop_img = img[y1:y2, x1:x2]
                text = run_ocr_on_crop(crop_img)
                
            new_tracked_plates.append({
                'box': cur_box,
                'text': text
            })
        else:
            # Biển số mới xuất hiện: chạy OCR ngay lập tức
            crop_img = img[y1:y2, x1:x2]
            text = run_ocr_on_crop(crop_img)
            
            new_tracked_plates.append({
                'box': cur_box,
                'text': text
            })
            
    # Vẽ khung và thông tin biển số đã định dạng
    for plate in new_tracked_plates:
        x1, y1, x2, y2 = plate['box']
        text = plate['text']
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    return annotated_img, new_tracked_plates

# ==========================================
# 4. XỬ LÝ VIDEO & WEBCAM TRỰC TIẾP
# ==========================================
def process_video(file_path):
    """Hàm xử lý file Video"""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở file Video!")
        return

    window_name = "NHAN DIEN BIEN SO - VIDEO (Bam 'q' de thoat)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    tracked_plates = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Xử lý khung hình có áp dụng tracking và giảm tải OCR
        result_frame, tracked_plates = process_frame_with_tracking(frame, frame_count, tracked_plates)
        
        # Co giãn kích thước hiển thị nếu quá lớn
        h, w = result_frame.shape[:2]
        if w > 1000 or h > 600:
            scale = min(1000/w, 600/h)
            result_frame = cv2.resize(result_frame, (int(w * scale), int(h * scale)))
            
        cv2.imshow(window_name, result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def process_realtime():
    """Hàm nhận diện thời gian thực qua Webcam (Chống trễ & Mờ)"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở Webcam!")
        return

    # Tối ưu hóa cấu hình camera để tăng chất lượng và độ mượt mà
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Đặt độ phân giải HD
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Giới hạn hàng đợi chỉ chứa 1 khung hình mới nhất

    window_name = "NHAN DIEN BIEN SO - WEBCAM TRUC TIEP (Bam 'q' de thoat)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    tracked_plates = []
    
    while True:
        # Đọc bỏ qua các khung hình cũ trong hàng đợi của phần cứng để đảm bảo hoàn toàn không bị trễ
        for _ in range(2):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break
            
        frame_count += 1
        
        # Xử lý với cơ chế tracking thông minh và chống mờ
        result_frame, tracked_plates = process_frame_with_tracking(frame, frame_count, tracked_plates)
        
        cv2.imshow(window_name, result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 5. GIAO DIỆN APP (UI)
# ==========================================
def process_file():
    """Hàm chọn và xử lý file Ảnh hoặc Video"""
    file_path = filedialog.askopenfilename(
        title="Chọn file Ảnh hoặc Video", 
        filetypes=[("Image/Video Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")]
    )
    if not file_path: 
        return

    # Kiểm tra nếu là file Video
    if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(file_path)
    else:
        # Xử lý file Ảnh
        img = cv2.imread(file_path)
        if img is None: 
            return
        
        result_img = process_single_image(img)
        
        # Co giãn kích thước hiển thị nếu quá lớn
        h, w = result_img.shape[:2]
        if w > 1200 or h > 700:
            scale = min(1200/w, 700/h)
            result_img = cv2.resize(result_img, (int(w * scale), int(h * scale)))
            
        cv2.imshow("APP NHAN DIEN BIEN SO - ANH", result_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

# Thiết lập Giao diện Tkinter
root = tk.Tk()
root.title("App AI Nhận Diện Biển Số Xe")
root.geometry("450x280")
root.configure(bg="#f0f4f8")
root.eval('tk::PlaceWindow . center')

# Header
tk.Label(
    root, 
    text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ", 
    font=("Helvetica", 16, "bold"), 
    fg="#1e3d59", 
    bg="#f0f4f8"
).pack(pady=25)

# Nút mở Webcam
btn_webcam = tk.Button(
    root, 
    text="📷 NHẬN DIỆN QUA WEBCAM", 
    command=process_realtime,
    font=("Helvetica", 12, "bold"),
    bg="#17b978",
    fg="white",
    activebackground="#118f5c",
    activeforeground="white",
    relief="flat",
    height=2
)
btn_webcam.pack(pady=8, fill='x', padx=50)

# Nút chọn File
btn_file = tk.Button(
    root, 
    text="📁 CHỌN FILE ẢNH / VIDEO", 
    command=process_file,
    font=("Helvetica", 12, "bold"),
    bg="#3f72af",
    fg="white",
    activebackground="#2b517e",
    activeforeground="white",
    relief="flat",
    height=2
)
btn_file.pack(pady=8, fill='x', padx=50)

# Chú thích chân trang
tk.Label(
    root, 
    text="Đồ án AI - Nhận diện & Định dạng biển số xe VN", 
    font=("Helvetica", 9, "italic"), 
    fg="#8b9ba8", 
    bg="#f0f4f8"
).pack(side="bottom", pady=10)

root.mainloop()