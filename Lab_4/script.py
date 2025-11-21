import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("as.png")

# ------------------------------
# 1. LAB для поиска серых полос
# ------------------------------
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

mask_gray = cv2.inRange(lab, (120, 120, 120), (245, 135, 135))
mask_gray_open = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

# ------------------------------
# 2. Маска текста
# ------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, text_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
protect = cv2.dilate(text_mask, np.ones((3,3), np.uint8), iterations=2)

# ------------------------------
# 3. Маска удаления полос
# ------------------------------
safe_mask = cv2.bitwise_and(mask_gray_open, cv2.bitwise_not(protect))

# ------------------------------
# 4. Очистка изображения
# ------------------------------
clean = img.copy()
clean[safe_mask == 255] = (255, 255, 255)

# ------------------------------
# 5. Бинаризация
# ------------------------------
gray2 = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
_, bin_text = cv2.threshold(gray2, 230, 255, cv2.THRESH_BINARY_INV)

bin_closed = cv2.morphologyEx(bin_text, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)

bin_final = cv2.bitwise_not(bin_closed)

# ------------------------------
# 6. Сохранение файлов
# ------------------------------
cv2.imwrite("result_clean.png", clean)
cv2.imwrite("result_binary.png", bin_final)

print("Готово: result_clean.png и result_binary.png")

# ------------------------------
# 7. PLOT со всеми этапами
# ------------------------------
plt.figure(figsize=(15,15))

titles = [
    "Original",
    "LAB L channel",
    "Gray Mask",
    "Gray Mask (Open)",
    "Text Mask",
    "Protect Mask",
    "Safe Mask",
    "Clean Image",
    "Gray2",
    "Binary INV",
    "Morph Close",
    "Final Binary"
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    L,
    mask_gray,
    mask_gray_open,
    text_mask,
    protect,
    safe_mask,
    cv2.cvtColor(clean, cv2.COLOR_BGR2RGB),
    gray2,
    bin_text,
    bin_closed,
    bin_final
]

for i, (t, im) in enumerate(zip(titles, images)):
    plt.subplot(4, 3, i+1)
    cmap = 'gray' if len(im.shape)==2 else None
    plt.imshow(im, cmap=cmap)
    plt.title(t)
    plt.axis('off')

plt.tight_layout()
plt.show()
