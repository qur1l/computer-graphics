from PIL import Image
import matplotlib.pyplot as plt

canvas_size = 500

def bresenham_line(x1, y1, x2, y2, image):
    """Алгоритм Брезенхема для построения линии"""
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

# Создание изображения
image = Image.new('RGB', (canvas_size, canvas_size))

# Координаты по умолчанию
start_x, start_y = 350, 60
end_x, end_y = 1, 320

# Запрос координат
print("Хотите ввести координаты? 1 - да, любой другой символ - нет")
choice = input()

if choice == "1":
    print("Введите 4 координаты (x1 y1 x2 y2): ")
    start_x, start_y, end_x, end_y = map(int, input().split())

# Рисование линии
bresenham_line(start_x, start_y, end_x, end_y, image)

# Показ изображения
plt.imshow(image)
plt.title(f'Алгоритм Брезенхема: ({start_x}, {start_y}) -> ({end_x}, {end_y})')
plt.axis('off')
plt.show()