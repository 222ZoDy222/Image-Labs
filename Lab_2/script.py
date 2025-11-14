import cv2
import numpy as np
import matplotlib.pyplot as plt


def manual_equalization(img: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = 255 * cdf / cdf[-1]
    lut = np.round(cdf_normalized).astype(np.uint8)
    return lut[img]


def apply_clahe(img: np.ndarray, clip_limit=2.0, grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def plot_results(original, manual, clahe):
    hist_original = cv2.calcHist([original], [0], None, [256], [0, 256])
    hist_manual = cv2.calcHist([manual], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe], [0], None, [256], [0, 256])

    plt.figure(figsize=(14, 10))
    plt.subplot(3, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.title('Оригинал')

    plt.subplot(3, 2, 2)
    plt.plot(hist_original, color='black')
    plt.title('Гистограмма оригинала')

    plt.subplot(3, 2, 3)
    plt.imshow(manual, cmap='gray')
    plt.axis('off')
    plt.title('Эквализация вручную')

    plt.subplot(3, 2, 4)
    plt.plot(hist_manual, color='black')
    plt.title('Гистограмма (вручную)')

    plt.subplot(3, 2, 5)
    plt.imshow(clahe, cmap='gray')
    plt.axis('off')
    plt.title('CLAHE (OpenCV)')

    plt.subplot(3, 2, 6)
    plt.plot(hist_clahe, color='black')
    plt.title('Гистограмма (CLAHE)')

    plt.tight_layout()
    plt.show()


def process_image(filename: str):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    manual = manual_equalization(img)
    clahe = apply_clahe(img)
    plot_results(img, manual, clahe)

process_image("lenna.png")
process_image("winter_cat.png")
