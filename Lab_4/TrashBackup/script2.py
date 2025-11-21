import cv2
import numpy as np

def clean_image(image_path, output_path):
    # 1. Загружаем изображение
    img = cv2.imread(image_path)
    
    if img is None:
        print("Ошибка: Изображение не найдено.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_value = 180 
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, binary)

# Запуск функции
# Замените 'input.png' на имя вашего файла
clean_image('as.png', 'cleaned_output.png')