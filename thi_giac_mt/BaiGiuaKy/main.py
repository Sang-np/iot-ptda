import cv2
import numpy as np

# 1. Khởi tạo
cap = cv2.VideoCapture('video.mp4')
# Giảm varThreshold (từ 40 xuống 16) để phát hiện xe nhỏ/xa hơn
detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)

# Biến phục vụ Tracking và Đếm
tracked_objects = {} # {id: (x, y)}
next_object_id = 0
total_counted = 0
counted_ids = set()


while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. Tiền xử lý ảnh (Làm sạch mặt nạ)
    mask = detector.apply(frame)
    # Loại bỏ bóng đổ (màu xám) chỉ giữ lại màu trắng (vật thể thật)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    # Dùng Dilate để nối các phần rời rạc của xe lại với nhau
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Hạ thấp diện tích tối thiểu xuống 300 để đếm được xe ở xa
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = int(x + w/2), int(y + h/2)
            current_centroids.append((x, y, w, h, cx, cy))

    # 3. Logic Tracking ID
    new_tracked_objects = {}
    for (x, y, w, h, cx, cy) in current_centroids:
        assigned = False
        for obj_id, old_centroid in tracked_objects.items():
            # Tính khoảng cách Euclidean giữa tâm cũ và tâm mới
            dist = np.hypot(cx - old_centroid[0], cy - old_centroid[1])
            if dist < 40: # Nếu gần nhau thì coi là cùng 1 xe
                new_tracked_objects[obj_id] = (cx, cy)
                assigned = True
                # Vẽ Box và ID
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{obj_id}", (x, y - 10), 0, 0.5, (0,255,0), 2)
                break
        
        if not assigned:
            new_tracked_objects[next_object_id] = (cx, cy)
            next_object_id += 1

    tracked_objects = new_tracked_objects
    
    # 4. Hiển thị thông tin
    cv2.putText(frame, f"Tong xe da xuat hien: {next_object_id}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tracking ID - OpenCV", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()