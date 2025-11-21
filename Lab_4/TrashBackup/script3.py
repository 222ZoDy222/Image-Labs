import cv2
import numpy as np

def clean_image(path_in, path_out):
    img = cv2.imread(path_in)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    kernel_er = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(opened, kernel_er, iterations=1)

    kernel_dil = np.ones((2, 2), np.uint8)
    restored = cv2.dilate(eroded, kernel_dil, iterations=1)

    cv2.imwrite(path_out, restored)

clean_image("as.png", "cleaned_output.png")
