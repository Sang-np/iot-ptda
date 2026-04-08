import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime

# ==========================================================
# 1. KHỞI TẠO VÀ CẤU HÌNH
# ==========================================================
# Tải mô hình YOLOv8 Nano (nhẹ và nhanh)
model = YOLO('./yolov8n.pt') 

# Cấu hình Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Tạo thư mục lưu trữ ảnh xâm nhập
output_folder = "intruders"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Định nghĩa vùng cấm (Tọa độ 4 góc của đa giác)
# Bạn có thể điều chỉnh các số này để thay đổi hình dáng vùng cấm
area_pts = np.array([[100, 200], [1180, 200], [1180, 700], [100, 700]], np.int32)

# Các biến hỗ trợ
prev_time = 0
last_saved_time = 0
save_cooldown = 3  # Giờ nghỉ giữa 2 lần lưu ảnh (3 giây)

def check_inside(center_point, polygon):
    """Kiểm tra một điểm có nằm trong đa giác không"""
    result = cv2.pointPolygonTest(polygon, center_point, False)
    return result >= 0

# ==========================================================
# 2. VÒNG LẶP XỬ LÝ CHÍNH
# ==========================================================
print("--- HỆ THỐNG AN NINH THÔNG MINH ĐANG CHẠY ---")
print("Nhấn 'q' để thoát chương trình.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Lật ảnh để dễ quan sát (như gương)
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Nhận diện đối tượng (chỉ lấy class 0 là 'người')
    results = model.predict(frame, conf=0.5, classes=0, verbose=False)

    # Vẽ vùng giám sát lên màn hình
    cv2.polylines(display_frame, [area_pts], isClosed=True, color=(255, 191, 0), thickness=2)
    cv2.putText(display_frame, "KHU VUC CAM XAM NHAP", (110, 180), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 191, 0), 2)

    intrusion_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Tọa độ khung bao
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Tính toán tâm phía dưới của đối tượng (vị trí đứng)
            cx, cy = int((x1 + x2) / 2), int(y2)

            # Kiểm tra xâm nhập
            if check_inside((cx, cy), area_pts):
                intrusion_detected = True
                # Vẽ khung màu đỏ cho kẻ xâm nhập
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.circle(display_frame, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(display_frame, "CANH BAO!", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Vẽ khung màu xanh cho người ở ngoài vùng cấm
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Xử lý lưu ảnh khi có xâm nhập
    if intrusion_detected:
        current_time = time.time()
        if current_time - last_saved_time > save_cooldown:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"intruder_{timestamp}.jpg"
            cv2.imwrite(os.path.join(output_folder, img_name), display_frame)
            print(f"⚠️ [CANH BAO] Đã lưu bằng chứng: {img_name}")
            last_saved_time = current_time

    # Tính toán và hiển thị FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Hiển thị cửa sổ
    cv2.imshow("Smart Intrusion Detection System", display_frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("--- HỆ THỐNG ĐÃ DỪNG ---")
