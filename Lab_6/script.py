import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb


image = cv.imread('./green.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Исходное изображение (green.jpg)")
plt.axis("off")
plt.show()


# сэмплинг

r, g, b = cv.split(image_rgb)


step = 10
r_s = r.flatten()[::step]
g_s = g.flatten()[::step]
b_s = b.flatten()[::step]

pixel_colors = image_rgb.reshape((-1, 3))[::step]

norm = colors.Normalize(vmin=0, vmax=255)
pixel_colors = norm(pixel_colors).tolist()

fig = plt.figure(figsize=(8, 8))
axis = fig.add_subplot(111, projection="3d")

axis.scatter(r_s, g_s, b_s,
             facecolors=pixel_colors,
             marker=".",
             s=1)

axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()



# Канал Green 
green = image_rgb.copy()
green[:, :, 0] = 0  # Red
green[:, :, 2] = 0  # Blue

plt.imshow(green)
plt.title("Зелёный канал (RGB)")
plt.axis("off")
plt.show()

# RGB в HSV
image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
h, s, v = cv.split(image_hsv)

step = 10
h_s = h.flatten()[::step]
s_s = s.flatten()[::step]
v_s = v.flatten()[::step]

h_s = h_s / 179.0
s_s = s_s / 255.0
v_s = v_s / 255.0

# Цвета точек — реальные RGB
pixel_colors = image_rgb.reshape((-1, 3))[::step] / 255.0

fig = plt.figure(figsize=(8, 8))
axis = fig.add_subplot(111, projection="3d")

axis.scatter(h_s, s_s, v_s,
             facecolors=pixel_colors,
             marker=".",
             s=1)

axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")

plt.show()

# Маска ЗЕЛЁНОГО (HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

lg_square = np.full((10, 10, 3), lower_green, dtype=np.uint8) / 255.0
ug_square = np.full((10, 10, 3), upper_green, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lg_square))
plt.title("Lower green")
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(ug_square))
plt.title("Upper green")
plt.show()

# Применение маски
mask_green = cv.inRange(image_hsv, lower_green, upper_green)
result_green = cv.bitwise_and(image_rgb, image_rgb, mask=mask_green)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Исходное")
plt.subplot(1, 3, 2)
plt.imshow(mask_green, cmap="gray")
plt.title("Маска")
plt.subplot(1, 3, 3)
plt.imshow(result_green)
plt.title("Зелёные области")
plt.show()

# Дополнительная маска
light_green = (35, 10, 180)
dark_green = (85, 120, 255)

mask_light = cv.inRange(image_hsv, light_green, dark_green)

final_mask = mask_green + mask_light
final_result = cv.bitwise_and(image_rgb, image_rgb, mask=final_mask)

blur = cv.GaussianBlur(final_result, (7, 7), 0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(final_mask, cmap="gray")
plt.title("Итоговая маска")
plt.subplot(1, 3, 2)
plt.imshow(final_result)
plt.title("Результат")
plt.subplot(1, 3, 3)
plt.imshow(blur)
plt.title("Сглаживание")
plt.show()

# Функция сегментации
def segment_image(image):

    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    lower_green = (35, 40, 40)
    upper_green = (85, 255, 255)

    mask_green = cv.inRange(hsv, lower_green, upper_green)

    light_green = (35, 10, 180)
    dark_green = (85, 120, 255)

    mask_light = cv.inRange(hsv, light_green, dark_green)

    final_mask = mask_green + mask_light
    result = cv.bitwise_and(image, image, mask=final_mask)

    blur = cv.GaussianBlur(result, (7, 7), 0)
    return blur

segmented = segment_image(image_rgb)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Исходное")
plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title("Сегментация зелёного")
plt.show()
