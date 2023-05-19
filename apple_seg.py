import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-img","--img_path", type=str, help="Path to the image")
args = parser.parse_args()

img_path = args.img_path
img_name = img_path.split("/")[-1]

image = cv2.imread(img_path)

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

# Apply morphological operations to improve mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Apply the mask to extract only the red regions
red_apple = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite(f"result/{img_name}", red_apple)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Count the number of objects
num_labels, labeled_image = cv2.connectedComponents(mask)

# Subtract 1 from the total count to exclude the background label
num_objects = int(num_labels)
print(f"{img_name}:", num_objects)

