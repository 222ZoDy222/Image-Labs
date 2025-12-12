import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Загружаем изображение
image = cv.imread("green.jpg")
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Переводим в HSV
hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

# Диапазон зелёного
lower_green = np.array([35, 50, 60])
upper_green = np.array([85, 255, 255])

# Создаем маску
mask = cv.inRange(hsv, lower_green, upper_green)

# Накладываем маску на изображение
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask)

# Сгладим результат
result_blur = cv.GaussianBlur(result, (7, 7), 0)

# ---- Нахождение контура и рамки ----

# Находим контуры на бинарной маске
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Делаем копию изображения для рисования
boxed = image_rgb.copy()

for cnt in contours:
    area = cv.contourArea(cnt)
    # Игнорируем слишком маленькие области
    if area < 500:
        continue
    x, y, w, h = cv.boundingRect(cnt)
    # Рисуем рамку
    cv.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 10)

# ------------------------------------

# Покажем результат
plt.figure(figsize=(18, 6))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask, cmap="gray")
plt.title("Маска зелёного")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(result_blur)
plt.title("Выделенная зелёная жидкость")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(boxed)
plt.title("Область выделена рамкой")
plt.axis("off")

plt.show()
