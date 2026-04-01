import cv2
import numpy as np

# 1. Khởi tạo Video và Bộ tách nền
cap = cv2.VideoCapture('video.mp4')
detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)

# --- THIẾT LẬP VẠCH KẺ VÀ ĐẾM ---
line_y = 400      # Vị trí chiều cao vạch kẻ (Thay đổi số này để di chuyển vạch)
offset = 7        # Khoảng sai số để bắt tâm xe chạm vạch
dem_tong = 0
da_dem_ids = set()
tracked_objects = {}
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Lấy kích thước frame để vẽ vạch full màn hình
    height, width, _ = frame.shape

    # 2. Tiền xử lý ảnh (Tạo mặt nạ chuyển động)
    mask = detector.apply(frame)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2) # Làm dày vật thể để dễ tìm contour
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ vạch kẻ ngang (Màu vàng)
    cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 0), 2)
    cv2.putText(frame, "VACH DEM XE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    new_tracked_objects = {}
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500: # Lọc nhiễu
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = int(x + w/2), int(y + h/2) # Tâm vật thể

            # 3. Logic Tracking ID (Gán ID dựa trên khoảng cách tâm)
            assigned_id = None
            for obj_id, pos in tracked_objects.items():
                dist = np.hypot(cx - pos[0], cy - pos[1])
                if dist < 45: # Nếu tâm gần tâm cũ trong khoảng 45px
                    assigned_id = obj_id
                    break
            
            if assigned_id is None:
                assigned_id = next_id
                next_id += 1
            
            new_tracked_objects[assigned_id] = (cx, cy)

            # 4. Logic Đếm khi xe chạm vạch
            if (line_y - offset) < cy < (line_y + offset):
                if assigned_id not in da_dem_ids:
                    dem_tong += 1
                    da_dem_ids.add(assigned_id)
                    # Hiệu ứng đổi màu vạch khi có xe chạm vào (nháy Đỏ)
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 5)

            # 5. Vẽ Bounding Box và ID
            # Nếu xe đã được đếm thì vẽ màu Xanh lá, chưa đếm thì màu Trắng
            color = (0, 255, 0) if assigned_id in da_dem_ids else (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1) # Vẽ tâm xe
            cv2.putText(frame, f"ID:{assigned_id}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    tracked_objects = new_tracked_objects

    # 6. Hiển thị tổng số lượng xe đã đếm
    cv2.rectangle(frame, (0, 0), (280, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"TONG XE: {dem_tong}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("DEM XE GIUA KY", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()