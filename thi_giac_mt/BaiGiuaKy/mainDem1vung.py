import cv2
import numpy as np

# 1. Khởi tạo
cap = cv2.VideoCapture('video.mp4')
detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25)

# --- THIẾT LẬP VÙNG ROI VÀ VẠCH KẺ ---
# Giả sử bạn chỉ muốn đếm ở nửa bên trái màn hình hoặc một làn đường cụ thể
roi_x_start = 100   # Giới hạn bên trái
roi_x_end = 800     # Giới hạn bên phải
roi_y_start = 200   # Giới hạn trên
roi_y_end = 600     # Giới hạn dưới

line_y = 450        # Vạch kẻ nằm bên trong vùng ROI
offset = 7
dem_tong = 0
da_dem_ids = set()
tracked_objects = {}
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    height, width, _ = frame.shape

    # 2. Tiền xử lý
    mask = detector.apply(frame)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ khung ROI để giảng viên thấy vùng đang xử lý (Màu xám nhạt)
    cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (100, 100, 100), 2)
    cv2.putText(frame, "VUNG ROI", (roi_x_start, roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Vẽ vạch kẻ (Chỉ vẽ trong phạm vi ROI cho đẹp)
    cv2.line(frame, (roi_x_start, line_y), (roi_x_end, line_y), (255, 255, 0), 2)

    new_tracked_objects = {}
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = int(x + w/2), int(y + h/2)

            # --- BƯỚC QUAN TRỌNG: KIỂM TRA XE CÓ NẰM TRONG ROI KHÔNG ---
            if roi_x_start < cx < roi_x_end and roi_y_start < cy < roi_y_end:
                
                # Logic Tracking ID
                assigned_id = None
                for obj_id, pos in tracked_objects.items():
                    dist = np.hypot(cx - pos[0], cy - pos[1])
                    if dist < 45:
                        assigned_id = obj_id
                        break
                
                if assigned_id is None:
                    assigned_id = next_id
                    next_id += 1
                
                new_tracked_objects[assigned_id] = (cx, cy)

                # Logic Đếm khi chạm vạch (Chỉ đếm nếu đã nằm trong ROI)
                if (line_y - offset) < cy < (line_y + offset):
                    if assigned_id not in da_dem_ids:
                        dem_tong += 1
                        da_dem_ids.add(assigned_id)

                # Vẽ Box cho những xe trong vùng ROI
                color = (0, 255, 0) if assigned_id in da_dem_ids else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, h + y), color, 2)
                cv2.putText(frame, f"ID:{assigned_id}", (x, y - 10), 0, 0.5, color, 2)

    tracked_objects = new_tracked_objects

    # 3. Hiển thị bảng điều khiển
    cv2.rectangle(frame, (0, 0), (280, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"ROI COUNT: {dem_tong}", (20, 45), 0, 1, (255, 255, 255), 2)

    cv2.imshow("Kiem tra - ROI Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()