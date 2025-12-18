from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def compute_gradients(image):
    """Вычисление градиентов изображения"""
    img_array = np.array(image, dtype=float)
    
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    # Оператор Собеля
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    height, width = img_array.shape
    Ix = np.zeros_like(img_array)
    Iy = np.zeros_like(img_array)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = img_array[i-1:i+2, j-1:j+2]
            Ix[i, j] = np.sum(window * sobel_x)
            Iy[i, j] = np.sum(window * sobel_y)
    
    return Ix, Iy

def harris_detector(image, k=0.04, threshold=0.01, window_size=3):
    """
    Детектор углов Харриса
    
    k: параметр чувствительности (обычно 0.04-0.06)
    threshold: порог для определения угла
    window_size: размер окна для гауссова сглаживания
    """
    # Вычисляем градиенты
    Ix, Iy = compute_gradients(image)
    
    # Вычисляем произведения градиентов
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # Применяем гауссово сглаживание
    sigma = window_size / 6
    Sx2 = gaussian_filter(Ix2, sigma)
    Sy2 = gaussian_filter(Iy2, sigma)
    Sxy = gaussian_filter(Ixy, sigma)
    
    # Вычисляем отклик Харриса
    # R = det(M) - k * trace(M)^2
    # det(M) = Sx2 * Sy2 - Sxy^2
    # trace(M) = Sx2 + Sy2
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    
    R = det_M - k * (trace_M ** 2)
    
    # Нормализация
    R_normalized = R / R.max() if R.max() > 0 else R
    
    # Находим углы (локальные максимумы выше порога)
    corners = []
    height, width = R_normalized.shape
    
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            if R_normalized[i, j] > threshold:
                # Проверяем, является ли точка локальным максимумом
                window = R_normalized[i-1:i+2, j-1:j+2]
                if R_normalized[i, j] == window.max():
                    corners.append((j, i, R_normalized[i, j]))
    
    return R_normalized, corners

def non_maximum_suppression(corners, min_distance=10):
    """Подавление немаксимумов - оставляем только сильные углы"""
    if len(corners) == 0:
        return []
    
    # Сортируем по силе отклика
    corners_sorted = sorted(corners, key=lambda x: x[2], reverse=True)
    
    filtered_corners = []
    
    for corner in corners_sorted:
        x, y, response = corner
        
        # Проверяем расстояние до уже выбранных углов
        too_close = False
        for fx, fy, _ in filtered_corners:
            distance = np.sqrt((x - fx)**2 + (y - fy)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            filtered_corners.append(corner)
    
    return filtered_corners

def visualize_harris(image, R, corners):
    """Визуализация результатов"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Исходное изображение
    if len(np.array(image).shape) == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    # Тепловая карта отклика Харриса
    im = axes[1].imshow(R, cmap='hot')
    axes[1].set_title('Отклик детектора Харриса')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Обнаруженные углы
    if len(np.array(image).shape) == 3:
        axes[2].imshow(image)
    else:
        axes[2].imshow(image, cmap='gray')
    
    if len(corners) > 0:
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        axes[2].plot(x_coords, y_coords, 'r+', markersize=10, markeredgewidth=2)
    
    axes[2].set_title(f'Обнаруженные углы ({len(corners)} точек)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    print("Детектор углов Харриса")
    print("=" * 60)
    
    # Загрузка изображения
    print("\nВведите путь к изображению (или Enter для тестового):")
    path = input().strip()
    
    if path == "":
        # Создаём тестовое изображение с углами
        img_array = np.ones((200, 200), dtype=np.uint8) * 128
        # Квадрат
        img_array[50:100, 50:100] = 255
        # Треугольник
        for i in range(50):
            img_array[120+i, 120:120+i] = 200
        # Крест
        img_array[140:160, 150:170] = 180
        img_array[130:170, 158:162] = 180
        
        image = Image.fromarray(img_array)
        print("Используется тестовое изображение")
    else:
        image = Image.open(path)
    
    # Параметры детектора
    print("\nИспользовать параметры по умолчанию? (Enter - да, 1 - настроить)")
    choice = input().strip()
    
    if choice == "1":
        print("Введите k (0.04-0.06, по умолчанию 0.04):")
        k = float(input().strip() or "0.04")
        print("Введите threshold (0.001-0.1, по умолчанию 0.01):")
        threshold = float(input().strip() or "0.01")
        print("Введите window_size (3-7, по умолчанию 3):")
        window_size = int(input().strip() or "3")
    else:
        k = 0.04
        threshold = 0.01
        window_size = 3
    
    print(f"\nПараметры: k={k}, threshold={threshold}, window_size={window_size}")
    print("Обработка...")
    
    # Применяем детектор Харриса
    R, corners = harris_detector(image, k=k, threshold=threshold, window_size=window_size)
    
    print(f"Найдено углов (до фильтрации): {len(corners)}")
    
    # Подавление немаксимумов
    corners_filtered = non_maximum_suppression(corners, min_distance=10)
    
    print(f"Найдено углов (после фильтрации): {len(corners_filtered)}")
    
    # Визуализация
    visualize_harris(image, R, corners_filtered)
    
    print("\n" + "=" * 60)
    print("Готово!")