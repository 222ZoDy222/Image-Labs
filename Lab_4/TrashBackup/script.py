import cv2
import numpy as np

img = cv2.imread('as.png')

mask_gray = cv2.inRange(img, (150,150,150), (225,225,225))

mask_gray = cv2.medianBlur(mask_gray, 5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, text_mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)

protect = cv2.dilate(text_mask, np.ones((3,3), np.uint8), iterations=2)

safe_mask = cv2.bitwise_and(mask_gray, cv2.bitwise_not(protect))

clean = cv2.inpaint(img, safe_mask, 7, cv2.INPAINT_TELEA)

cv2.imwrite('result.png', clean)
print("Saved result.png")
