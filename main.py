import cv2
import numpy as np

# Kích thước bàn cờ
board_size = 8
cell_size = 80

# Tạo ảnh đen
img_size = board_size * cell_size
img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Vẽ bàn cờ
for row in range(board_size):
    for col in range(board_size):
        # Nếu (row + col) chẵn → ô trắng
        if (row + col) % 2 == 0:
            color = (255, 255, 255)  # trắng
        else:
            color = (0, 0, 0)        # đen

        # Tọa độ góc trên trái và dưới phải
        top_left = (col * cell_size, row * cell_size)
        bottom_right = ((col + 1) * cell_size, (row + 1) * cell_size)

        # Vẽ hình chữ nhật
        cv2.rectangle(img, top_left, bottom_right, color, -1)
        

# Hiển thị ảnh
cv2.imshow("Chessboard", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
