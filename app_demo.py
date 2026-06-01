import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import RTDETR

print("Đang tải mô hình RT-DETR, vui lòng đợi...")
model = RTDETR('best.pt') 
print("Tải mô hình thành công!")

def process_realtime():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở Webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=0.4) 
        annotated_frame = results[0].plot()
        cv2.imshow("Webcam (Bam 'q' de thoat)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

def process_file():
    file_path = filedialog.askopenfilename(title="Chọn file", filetypes=[("Image/Video", "*.jpg *.jpeg *.png *.mp4")])
    if not file_path: return

    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(file_path)
        results = model(img, conf=0.4)
        annotated_img = results[0].plot()
        
        h, w = annotated_img.shape[:2]
        if w > 1200 or h > 700:
            scale = min(1200/w, 700/h)
            annotated_img = cv2.resize(annotated_img, (int(w * scale), int(h * scale)))
            
        cv2.imshow("Ket qua - Bam phim bat ky de dong", annotated_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
root = tk.Tk()
root.title("App Nhận Diện Biển Số Xe")
root.geometry("400x250") 
root.eval('tk::PlaceWindow . center') 
tk.Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ", font=("Arial", 16, "bold"), fg="blue").pack(pady=20)
tk.Button(root, text="1. Nhận diện qua Webcam", font=("Arial", 12), bg="lightgreen", command=process_realtime).pack(pady=10, fill='x', padx=40)
tk.Button(root, text="2. Chọn file Ảnh / Video", font=("Arial", 12), bg="lightblue", command=process_file).pack(pady=10, fill='x', padx=40)
root.mainloop()