import sys
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


img = cv.imread('img.jpg')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

denoise_median = cv.medianBlur(img, 5)

denoise = cv.bilateralFilter(denoise_median, d=9, sigmaColor=75, sigmaSpace=75)

lab = cv.cvtColor(denoise, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
l2 = clahe.apply(l)

lab2 = cv.merge((l2, a, b))
bright = cv.cvtColor(lab2, cv.COLOR_LAB2BGR)

blur = cv.GaussianBlur(bright, (0, 0), 3)
sharp = cv.addWeighted(bright, 1.6, blur, -0.6, 0)

def fix_line_shift(img, shift_per_row=1):
    h, w, c = img.shape
    corrected = np.zeros_like(img)

    for y in range(h):
        shift = (y * shift_per_row) % w   # насколько сдвигаем строку обратно
        corrected[y] = np.roll(img[y], -shift, axis=0)

    return corrected

corrected = fix_line_shift(sharp, shift_per_row=0.5)


ROI = (340, 250, 0, 0)  #(x0, y0, height, width)
cropped_image = corrected[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]


plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1), plt.imshow(rgb), plt.title("Исходная")
plt.subplot(2, 3, 2), plt.imshow(cv.cvtColor(denoise, cv.COLOR_BGR2RGB)), plt.title("Удаление шума")
plt.subplot(2, 3, 3), plt.imshow(cv.cvtColor(bright, cv.COLOR_BGR2RGB)), plt.title("CLAHE")
plt.subplot(2, 3, 4), plt.imshow(cv.cvtColor(sharp, cv.COLOR_BGR2RGB)), plt.title("Резкость")
plt.subplot(2, 3, 5), plt.imshow(cv.cvtColor(corrected, cv.COLOR_BGR2RGB)), plt.title("Исправлен сдвиг строк")
plt.tight_layout()
plt.show()
