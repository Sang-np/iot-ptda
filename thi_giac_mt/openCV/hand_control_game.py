import cv2
import mediapipe as mp
import keyboard
import time

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# CAMERA SETUP
# =========================

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

# =========================
# GAME CONTROL
# =========================

last_jump = 0
jump_delay = 0.25
mode_text = "WAITING"

# =========================
# FPS
# =========================

pTime = 0

# =========================
# COUNT FINGERS
# =========================

def count_fingers(hand_landmarks):

    fingers = 0

    tips = [8,12,16,20]

    for tip in tips:

        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            fingers += 1

    return fingers


# =========================
# START MESSAGE
# =========================

print("===================================")
print("FLAPPY BIRD AI CONTROL")
print("===================================")
print("☝ One finger -> Jump")
print("✊ No finger -> No jump")
print("Press Q to quit")
print("===================================")


# =========================
# MAIN LOOP
# =========================

while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img,1)

    # Convert màu
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    imgRGB.flags.writeable = False
    results = hands.process(imgRGB)
    imgRGB.flags.writeable = True

    current_time = time.time()

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            # =========================
            # DRAW HAND LANDMARKS
            # =========================

            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            # =========================
            # COUNT FINGERS
            # =========================

            fingers = count_fingers(handLms)

            # =========================
            # GAME CONTROL
            # =========================

            if fingers == 1:

                mode_text = "JUMP"

                if current_time - last_jump > jump_delay:

                    keyboard.press_and_release("space")
                    last_jump = current_time

            else:

                mode_text = "NO JUMP"

    else:

        mode_text = "NO HAND"


    # =========================
    # FPS CALCULATION
    # =========================

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    # =========================
    # UI TEXT
    # =========================

    cv2.putText(
        img,
        f"MODE: {mode_text}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        img,
        f"FPS: {int(fps)}",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    # =========================
    # SHOW CAMERA
    # =========================

    cv2.imshow("Flappy Bird AI Control",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()