## proyecto final: ADAS
## por: Luis Padilla y Erick Duque

import cv2
import numpy as np

# --- Función de preprocesamiento ---
def preprocesar_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# --- Aplicar máscara para excluir áreas no relevantes (tablero del vehículo) ---
def aplicar_mascara(frame):
    height, width = frame.shape[:2]
    mask = np.zeros_like(frame)

    # Definir la región de interés (ROI) que excluye el tablero
    polygon = np.array([[
        (0, height * 0.55),  # Izquierda superior
        (width, height * 0.55),  # Derecha superior
        (width, height),  # Derecha inferior
        (0, height)  # Izquierda inferior
    ]], np.int32)

    # Crear una máscara inversa
    cv2.fillPoly(mask, polygon, 255)
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame

# --- Función para detectar vehículos ---
def detectar_vehiculos(frame, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detected_vehicles = []
    height, width = frame.shape[:2]

    # Filtrar contornos
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Reducir el área mínima para detectar vehículos más lejanos
            x, y, w, h = cv2.boundingRect(contour)
            # Validar el tamaño y la relación de aspecto
            aspect_ratio = w / float(h)
            if aspect_ratio > 0.5 and aspect_ratio < 3.0:
                if height * 0.2 < y < height * 0.6:  # Limitar detección a la zona de la carretera
                    detected_vehicles.append((x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Rojo para vehículos
    return frame, detected_vehicles

# --- Función para detectar líneas de carril ---
def detectar_lineas(frame_prep):
    edges = cv2.Canny(frame_prep, 50, 150)  # Ajustar los umbrales del Canny

    # Definir región de interés (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    # Transformada de Hough para detectar líneas
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=150)

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 999
            intercept = y1 - slope * x1

            # Filtrar líneas por pendiente y posición
            if 0.5 < abs(slope) < 2:  # Evitar líneas horizontales o verticales
                if slope < 0 and x1 < width // 2 and x2 < width // 2:
                    left_lines.append((slope, intercept))
                elif slope > 0 and x1 > width // 2 and x2 > width // 2:
                    right_lines.append((slope, intercept))

    return left_lines, right_lines

# --- Función para promediar las líneas detectadas ---
def promediar_lineas(frame, lines):
    if not lines:
        return None

    slope_avg = np.mean([line[0] for line in lines])
    intercept_avg = np.mean([line[1] for line in lines])

    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)
    x1 = int((y1 - intercept_avg) / slope_avg)
    x2 = int((y2 - intercept_avg) / slope_avg)

    return (x1, y1, x2, y2)

# --- Función para dibujar las líneas promedio ---
def dibujar_lineas(frame, left_lines, right_lines):
    left_lane = promediar_lineas(frame, left_lines)
    right_lane = promediar_lineas(frame, right_lines)

    if left_lane is not None:
        x1, y1, x2, y2 = left_lane
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Azul para carril izquierdo

    if right_lane is not None:
        x1, y1, x2, y2 = right_lane
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Verde para carril derecho

    return frame

# --- Función principal para procesar el video ---
def procesar_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar frame
        frame_prep = preprocesar_frame(frame)

        # Aplicar máscara para eliminar tablero y ruido
        frame_prep = aplicar_mascara(frame_prep)

        # Detectar líneas de carril
        left_lines, right_lines = detectar_lineas(frame_prep)
        frame = dibujar_lineas(frame, left_lines, right_lines)

        # Detectar vehículos
        edges = cv2.Canny(frame_prep, 50, 150)  # Umbrales más bajos
        frame, vehiculos = detectar_vehiculos(frame, edges)

        # Mostrar resultados
        window_name = "Detección de Carril y Vehículos"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# --- Ejecutar el programa ---
video_path = "NO20240818-165143-011993F.MP4" 
procesar_video(video_path)
