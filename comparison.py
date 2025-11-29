from PIL import Image
import matplotlib.pyplot as plt
import time

canvas_size = 500

def bresenham_integer(x1, y1, x2, y2, image):
    """Целочисленный метод Брезенхема"""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    error = dx - dy
    
    while True:
        if 0 <= x1 < canvas_size and 0 <= y1 < canvas_size:
            image.putpixel((y1, x1), (255, 255, 255))
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * error
        
        if e2 > -dy:
            error -= dy
            x1 += sx
        if e2 < dx:
            error += dx
            y1 += sy

def dda_float(x1, y1, x2, y2, image):
    """Вещественный метод (DDA)"""
    dx = x2 - x1
    dy = y2 - y1
    
    steps = max(abs(dx), abs(dy))
    
    if steps == 0:
        if 0 <= x1 < canvas_size and 0 <= y1 < canvas_size:
            image.putpixel((y1, x1), (255, 255, 255))
        return
    
    x_inc = dx / steps
    y_inc = dy / steps
    
    x = float(x1)
    y = float(y1)
    
    for _ in range(int(steps) + 1):
        xi = round(x)
        yi = round(y)
        if 0 <= xi < canvas_size and 0 <= yi < canvas_size:
            image.putpixel((yi, xi), (255, 255, 255))
        x += x_inc
        y += y_inc

def benchmark(method, x1, y1, x2, y2, iterations=1000):
    """Замер времени выполнения"""
    start = time.time()
    for _ in range(iterations):
        img = Image.new('RGB', (canvas_size, canvas_size))
        method(x1, y1, x2, y2, img)
    return time.time() - start

# Координаты по умолчанию
start_x, start_y = 350, 60
end_x, end_y = 1, 320

# Запрос координат
print("Хотите ввести координаты? 1 - да, любой другой символ - нет")
choice = input()

if choice == "1":
    print("Введите 4 координаты (x1 y1 x2 y2): ")
    start_x, start_y, end_x, end_y = map(int, input().split())

# Создание изображений
image_int = Image.new('RGB', (canvas_size, canvas_size))
image_float = Image.new('RGB', (canvas_size, canvas_size))

bresenham_integer(start_x, start_y, end_x, end_y, image_int)
dda_float(start_x, start_y, end_x, end_y, image_float)

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(image_int)
ax1.set_title('Целочисленный метод (Брезенхем)')
ax1.axis('off')

ax2.imshow(image_float)
ax2.set_title('Вещественный метод (DDA)')
ax2.axis('off')

plt.tight_layout()
plt.show()

# Сравнение производительности
print("\n" + "="*50)
print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ (1000 итераций)")
print("="*50)

time_int = benchmark(bresenham_integer, start_x, start_y, end_x, end_y)
time_float = benchmark(dda_float, start_x, start_y, end_x, end_y)

print(f"Целочисленный метод: {time_int:.4f} сек ({time_int*1000:.2f} мс)")
print(f"Вещественный метод:  {time_float:.4f} сек ({time_float*1000:.2f} мс)")
print(f"\nРазница: {time_float/time_int:.2f}x")
print(f"Целочисленный метод быстрее на {((time_float-time_int)/time_float*100):.1f}%")
print("="*50)