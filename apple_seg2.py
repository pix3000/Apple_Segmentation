import cv2
import numpy as np

img_name = "test/A08F.JPG"

image = cv2.imread(img_name)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Red color range definition
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create masks for red regions
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Apply the mask to extract only the red regions
red_apple = cv2.bitwise_and(image, image, mask=mask)

# Count the number of objects
num_labels, labeled_image = cv2.connectedComponents(mask)

# Subtract 1 from the total count to exclude the background label
num_objects = int(num_labels /45) 
print(f"{img_name}:", num_objects)

cv2.imwrite(f"result/{img_name}", red_apple)
cv2.waitKey(0)
cv2.destroyAllWindows()
