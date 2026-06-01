import cv2
import numpy as np
import os
import re
import logging
import time
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from ultralytics import RTDETR
from paddleocr import PaddleOCR
import threading

# ==========================================
# 1. KHỞI TẠO CÁC MÔ HÌNH AI & FLASK
# ==========================================
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Đang khởi động hệ thống AI cho máy chủ Web...")
print("- Đang tải RT-DETR...")
det_model = RTDETR('best.pt') 

logging.getLogger("ppocr").setLevel(logging.ERROR)
print("- Đang tải SVTR...")
ocr_model = PaddleOCR(use_textline_orientation=False, lang='en', use_mkldnn=False)
print("Hệ thống AI sẵn sàng!\n")

# Biến lưu trữ biển số đã nhận dạng để hiển thị lên Sidebar
latest_plates = []
plates_lock = threading.Lock()

# Quản lý webcam stream
webcam_cap = None
is_webcam_running = False
webcam_lock = threading.Lock()

# ==========================================
# 2. CÁC HÀM HẬU XỬ LÝ & TIỀN XỬ LÝ (TỪ APP_MAIN)
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

def get_iou(boxA, boxB):
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
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    crop_img = cv2.filter2D(crop_img, -1, kernel)
    return crop_img

def run_ocr_on_crop(crop_img):
    processed = preprocess_plate(crop_img)
    ocr_result = ocr_model.ocr(processed)
    text_raw = ""
    if ocr_result and ocr_result[0]:
        for res in ocr_result[0]:
            text_raw += res[1][0]
    final_text = hau_xu_ly(text_raw)
    return format_bien_so(final_text)

def record_detected_plate(text, source_type="Webcam"):
    """Ghi nhận lại biển số mới được phát hiện vào danh sách log"""
    global latest_plates
    if not text:
        return
    now_str = datetime.now().strftime("%H:%M:%S")
    with plates_lock:
        # Tránh ghi đè trùng lắp biển số trong cùng một giây
        if not any(p['text'] == text and p['time'] == now_str for p in latest_plates):
            latest_plates.append({
                'text': text,
                'time': now_str,
                'type': source_type
            })
            # Giới hạn danh sách tối đa 50 biển gần nhất
            if len(latest_plates) > 50:
                latest_plates.pop(0)

# ==========================================
# 3. QUY TRÌNH XỬ LÝ KHUNG HÌNH (VIDEO/WEBCAM)
# ==========================================
def process_frame(img, frame_count, tracked_plates, source_type="Webcam"):
    results = det_model(img, conf=0.4, verbose=False)
    annotated_img = img.copy()
    
    current_boxes = results[0].boxes.xyxy.cpu().numpy()
    new_tracked_plates = []
    
    for cur_box in current_boxes:
        cur_box = list(map(int, cur_box))
        x1, y1, x2, y2 = cur_box
        
        best_match = None
        best_iou = 0.25
        
        for plate in tracked_plates:
            iou = get_iou(cur_box, plate['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = plate
                
        if best_match is not None:
            text = best_match['text']
            if frame_count % 10 == 0 or not text:
                crop_img = img[y1:y2, x1:x2]
                text = run_ocr_on_crop(crop_img)
                if text:
                    record_detected_plate(text, source_type)
            new_tracked_plates.append({
                'box': cur_box,
                'text': text
            })
        else:
            crop_img = img[y1:y2, x1:x2]
            text = run_ocr_on_crop(crop_img)
            if text:
                record_detected_plate(text, source_type)
            new_tracked_plates.append({
                'box': cur_box,
                'text': text
            })
            
    for plate in new_tracked_plates:
        x1, y1, x2, y2 = plate['box']
        text = plate['text']
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(annotated_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
    return annotated_img, new_tracked_plates

# ==========================================
# 4. CÁC ROUTE CỦA FLASK SERVER
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def generate_webcam_stream():
    global webcam_cap, is_webcam_running
    
    with webcam_lock:
        if webcam_cap is None:
            webcam_cap = cv2.VideoCapture(0)
            webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            webcam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_webcam_running = True

    frame_count = 0
    tracked_plates = []
    
    while is_webcam_running:
        for _ in range(2):
            webcam_cap.grab()
        ret, frame = webcam_cap.retrieve()
        if not ret:
            break
            
        frame_count += 1
        annotated_frame, tracked_plates = process_frame(frame, frame_count, tracked_plates, "Webcam")
        
        # Mã hóa khung hình sang dạng JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Giải phóng camera khi kết thúc stream
    with webcam_lock:
        if webcam_cap is not None:
            webcam_cap.release()
            webcam_cap = None
        is_webcam_running = False

@app.route('/video_feed')
def video_feed():
    return Response(generate_webcam_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global is_webcam_running
    is_webcam_running = False
    return jsonify({"status": "success", "message": "Stream stopped"})

def generate_video_file_stream(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_count = 0
    tracked_plates = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Giảm tải xử lý video
        annotated_frame, tracked_plates = process_frame(frame, frame_count, tracked_plates, "Video File")
        
        # Co giãn kích thước về dạng nhỏ hơn trước khi truyền qua mạng
        h, w = annotated_frame.shape[:2]
        if w > 960 or h > 540:
            scale = min(960/w, 540/h)
            annotated_frame = cv2.resize(annotated_frame, (int(w * scale), int(h * scale)))
            
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/video_feed_file')
def video_feed_file():
    filepath = request.args.get('path', '')
    if not filepath or not os.path.exists(filepath):
        return "File not found", 404
    return Response(generate_video_file_stream(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"})
        
    # Lưu file
    filename = secure_filename_slug(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Kiểm tra là video hay ảnh
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({
            "status": "success",
            "type": "video",
            "filepath": filepath
        })
    else:
        # Xử lý file ảnh tĩnh
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"status": "error", "message": "Invalid image file"})
            
        # Nhận diện
        results = det_model(img, conf=0.4, verbose=False)
        annotated_img = img.copy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        detected_here = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            
            # Tiền xử lý + chạy OCR
            formatted_text = run_ocr_on_crop(crop_img)
            
            if formatted_text:
                record_detected_plate(formatted_text, "Upload Ảnh")
                now_str = datetime.now().strftime("%H:%M:%S")
                detected_here.append({
                    'text': formatted_text,
                    'time': now_str,
                    'type': "Upload Ảnh"
                })
                
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_img, formatted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(annotated_img, formatted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        # Lưu kết quả ảnh tĩnh đã vẽ khung
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, annotated_img)
        
        return jsonify({
            "status": "success",
            "type": "image",
            "result_url": f"/static/uploads/{result_filename}",
            "plates": detected_here
        })

@app.route('/get_latest_plates')
def get_latest_plates():
    with plates_lock:
        # Lấy bản sao danh sách biển số hiện tại và trả về
        return jsonify(list(latest_plates))

def secure_filename_slug(filename):
    # Tạo tên file an toàn
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return f"{name}_{int(time.time())}{ext}"

# ==========================================
# 5. KHỞI CHẠY SERVER
# ==========================================
if __name__ == '__main__':
    # Chạy trên localhost cổng 5000
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
