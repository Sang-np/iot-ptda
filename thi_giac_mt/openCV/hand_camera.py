import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Start program")

# Đường dẫn model
model_path = "hand_landmarker.task"

# Cấu hình model
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

# Tạo detector
detector = vision.HandLandmarker.create_from_options(options)

# Mở camera
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot open camera")
        break

    # Lật ảnh
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tạo ảnh mediapipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Nhận diện bàn tay
    result = detector.detect(mp_image)

    h, w, _ = frame.shape

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            for lm in hand_landmarks:

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()