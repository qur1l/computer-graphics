import cv2
import numpy as np

# Загрузка изображения с текстом и шаблона буквы "k"
src_img = cv2.imread('text.jpg')
k_template = cv2.imread('k_template.jpg', cv2.IMREAD_GRAYSCALE)

# Бинаризация шаблона (буква — белая)
_, k_template = cv2.threshold(
    k_template, 120, 255, cv2.THRESH_BINARY_INV
)

# Предобработка изображения
gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
# Слегка сгладим, чтобы контуры не дробились
gray = cv2.medianBlur(gray, 3)
_, binary = cv2.threshold(
    gray, 120, 255, cv2.THRESH_BINARY_INV
)

# Поиск контуров отдельных символов
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Сортировка слева направо
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

detected_k = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Фильтрация шума (ослаблена, чтобы не отрезать маленькие K)
    if w < 8 or h < 16:
        continue

    # Вырезаем символ и масштабируем под шаблон
    roi = binary[y:y+h, x:x+w]
    roi = cv2.resize(
        roi,
        (k_template.shape[1], k_template.shape[0])
    )

    # Сравнение с шаблоном
    response = cv2.matchTemplate(
        roi,
        k_template,
        cv2.TM_CCOEFF_NORMED
    )
    score = np.max(response)

    # Порог совпадения (понижен, чтобы не пропускать последнюю K)
    if score > 0.25:
        detected_k.append((x, y, w, h))

        cv2.rectangle(
            src_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(
            src_img,
            'k',
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

print(f"Найдено букв 'k': {len(detected_k)}")

cv2.imshow("Detected letter K", src_img)
cv2.waitKey(0)
cv2.destroyAllWindows()