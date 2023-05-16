import cv2
import numpy as np

img_path = "test/apple_3.JPG"
img_name = img_path.split("/")[1]

image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 빨간색 범위 정의
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# 이미지를 HSV로 변환합니다.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색 영역을 마스크합니다.
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 마스크를 적용하여 빨간색 부분만 추출합니다.
red_apple = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite(f"result/{img_name}", red_apple)
cv2.waitKey(0)
cv2.destroyAllWindows()
