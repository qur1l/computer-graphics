from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def average_filter(image, kernel_size=3):
    """Усредняющий фильтр"""
    img_array = np.array(image)
    height, width, channels = img_array.shape
    
    result = np.zeros_like(img_array)
    offset = kernel_size // 2
    
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            for c in range(channels):
                # Берём окрестность и усредняем
                window = img_array[i-offset:i+offset+1, j-offset:j+offset+1, c]
                result[i, j, c] = np.mean(window)
    
    return Image.fromarray(result.astype('uint8'))

def gaussian_filter(image, kernel_size=3):
    """Гауссов фильтр"""
    # Гауссово ядро
    if kernel_size == 3:
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 16
    elif kernel_size == 5:
        kernel = np.array([[1, 4, 6, 4, 1],
                          [4, 16, 24, 16, 4],
                          [6, 24, 36, 24, 6],
                          [4, 16, 24, 16, 4],
                          [1, 4, 6, 4, 1]]) / 256
    else:
        # Создаём гауссово ядро
        sigma = kernel_size / 6
        ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
    
    img_array = np.array(image)
    height, width, channels = img_array.shape
    
    result = np.zeros_like(img_array, dtype=float)
    offset = kernel_size // 2
    
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            for c in range(channels):
                # Применяем ядро
                window = img_array[i-offset:i+offset+1, j-offset:j+offset+1, c]
                result[i, j, c] = np.sum(window * kernel)
    
    return Image.fromarray(result.astype('uint8'))

# Загрузка изображения
print("Введите путь к изображению (или нажмите Enter для тестового изображения):")
path = input().strip()

if path == "":
    # Создаём тестовое изображение с шумом
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    image = Image.fromarray(img)
    print("Используется тестовое изображение с шумом")
else:
    image = Image.open(path)

# Применение фильтров
print("Применение фильтров...")
avg_result = average_filter(image, kernel_size=3)
gauss_result = gaussian_filter(image, kernel_size=3)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Исходное изображение')
axes[0].axis('off')

axes[1].imshow(avg_result)
axes[1].set_title('Усредняющий фильтр (3x3)')
axes[1].axis('off')

axes[2].imshow(gauss_result)
axes[2].set_title('Гауссов фильтр (3x3)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("Готово!")