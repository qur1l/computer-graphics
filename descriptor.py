from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def compute_gradients(image):
    """Вычисление градиентов изображения"""
    img_array = np.array(image, dtype=float)
    
    # Преобразуем в градации серого если цветное
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)
    
    # Градиенты по Собелю
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    height, width = img_array.shape
    grad_x = np.zeros_like(img_array)
    grad_y = np.zeros_like(img_array)
    
    # Свёртка для вычисления градиентов
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = img_array[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(window * sobel_x)
            grad_y[i, j] = np.sum(window * sobel_y)
    
    # Величина и направление градиента
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)
    
    return magnitude, angle

def bilinear_interpolate_gradient(magnitude, angle, x, y):
    """Билинейная интерполяция градиента в точке (x, y)"""
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = x1 + 1, y1 + 1
    
    # Проверка границ
    if x1 < 0 or y1 < 0 or x2 >= magnitude.shape[1] or y2 >= magnitude.shape[0]:
        return 0, 0
    
    # Веса для интерполяции
    wx = x - x1
    wy = y - y1
    
    # Интерполяция величины
    mag = (1 - wx) * (1 - wy) * magnitude[y1, x1] + \
          wx * (1 - wy) * magnitude[y1, x2] + \
          (1 - wx) * wy * magnitude[y2, x1] + \
          wx * wy * magnitude[y2, x2]
    
    # Интерполяция угла
    ang = (1 - wx) * (1 - wy) * angle[y1, x1] + \
          wx * (1 - wy) * angle[y1, x2] + \
          (1 - wx) * wy * angle[y2, x1] + \
          wx * wy * angle[y2, x2]
    
    return mag, ang

def compute_descriptor(magnitude, angle, keypoint_x, keypoint_y, grid_size=16):
    """
    Вычисление дескриптора для ключевой точки
    
    grid_size: размер сетки вокруг точки (16x16 для окрестности 17x17)
    """
    # Для сетки 16x16 берём окрестность 17x17
    window_size = grid_size + 1
    half_window = window_size // 2
    
    descriptor = []
    
    # Проходим по сетке 4x4 ячейки по 4x4 пикселя
    cell_size = 4
    cells_per_side = grid_size // cell_size
    
    for cell_i in range(cells_per_side):
        for cell_j in range(cells_per_side):
            # Гистограмма направлений для ячейки (8 бинов)
            histogram = np.zeros(8)
            
            # Проходим по всем пикселям в ячейке 4x4
            for i in range(cell_size):
                for j in range(cell_size):
                    # Координаты в окрестности 17x17
                    local_y = cell_i * cell_size + i
                    local_x = cell_j * cell_size + j
                    
                    # Глобальные координаты
                    y = keypoint_y - half_window + local_y
                    x = keypoint_x - half_window + local_x
                    
                    # Билинейная интерполяция градиента
                    mag, ang = bilinear_interpolate_gradient(magnitude, angle, x, y)
                    
                    # Определяем бин гистограммы (8 направлений)
                    bin_index = int((ang + np.pi) / (2 * np.pi) * 8) % 8
                    histogram[bin_index] += mag
            
            descriptor.extend(histogram)
    
    # Нормализация дескриптора
    descriptor = np.array(descriptor)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor

def visualize_descriptor(image, keypoint_x, keypoint_y, descriptor):
    """Визуализация дескриптора"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Показываем изображение с ключевой точкой
    ax1.imshow(image, cmap='gray')
    ax1.plot(keypoint_x, keypoint_y, 'ro', markersize=10)
    
    # Рисуем окрестность 17x17
    rect = plt.Rectangle((keypoint_x - 8.5, keypoint_y - 8.5), 17, 17, 
                         fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)
    ax1.set_title(f'Ключевая точка ({keypoint_x}, {keypoint_y})')
    ax1.axis('off')
    
    # Показываем дескриптор как гистограмму
    ax2.bar(range(len(descriptor)), descriptor)
    ax2.set_title(f'Дескриптор (размерность: {len(descriptor)})')
    ax2.set_xlabel('Индекс')
    ax2.set_ylabel('Значение')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Основная программа
if __name__ == "__main__":
    print("Дескриптор ключевых точек с билинейной интерполяцией")
    print("=" * 60)
    
    # Загрузка изображения
    print("\nВведите путь к изображению (или Enter для тестового):")
    path = input().strip()
    
    if path == "":
        # Создаём тестовое изображение
        img_array = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        # Добавляем несколько структурных элементов
        img_array[40:60, 40:60] = 255
        img_array[30:35, 30:70] = 100
        image = Image.fromarray(img_array)
        print("Используется тестовое изображение")
    else:
        image = Image.open(path).convert('L')
    
    # Ввод координат ключевой точки
    print("\nВведите координаты ключевой точки (x y):")
    print("(или Enter для центра изображения)")
    coords = input().strip()
    
    if coords == "":
        keypoint_x = image.width // 2
        keypoint_y = image.height // 2
    else:
        keypoint_x, keypoint_y = map(int, coords.split())
    
    print(f"\nВычисление дескриптора для точки ({keypoint_x}, {keypoint_y})...")
    
    # Вычисление градиентов
    magnitude, angle = compute_gradients(image)
    
    # Вычисление дескриптора
    descriptor = compute_descriptor(magnitude, angle, keypoint_x, keypoint_y)
    
    print(f"\nДескриптор вычислен!")
    print(f"Размерность: {len(descriptor)}")
    print(f"Первые 10 значений: {descriptor[:10]}")
    
    # Визуализация
    visualize_descriptor(image, keypoint_x, keypoint_y, descriptor)
    
    print("\n" + "=" * 60)
    print("Готово!")