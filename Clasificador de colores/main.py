import cv2
import numpy as np
import joblib

# Cargar el modelo entrenado y el codificador de etiquetas
knn = joblib.load('color_clasificador_modelo.pkl')
label_encoder_classes = np.load('etiqueta_encoder.npy')

cap = cv2.VideoCapture(0)

# Definir rangos de colores en HSV para rojo, azul y verde
color_ranges = {
    'rojo': ([0, 120, 70], [10, 255, 255]),
    'azul': ([110, 50, 50], [130, 255, 255]),
    'verde': ([40, 40, 40], [70, 255, 255])
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Recorrer los colores definidos en el diccionario
    for color_name, (lower, upper) in color_ranges.items():
        # Crear máscara para el color
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Aplicar detección de contornos sobre la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filtrar contornos pequeños
            if cv2.contourArea(contour) > 500:  # Ajustar el tamaño mínimo de contornos si es necesario
                # Obtener el rectángulo delimitador del contorno
                x, y, w, h = cv2.boundingRect(contour)

                # Dibujar un rectángulo alrededor del objeto detectado
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extraer la región de interés (ROI)
                roi = frame[y:y + h, x:x + w]

                # Redimensionar la ROI para que coincida con el tamaño de entrada del modelo
                roi_resized = cv2.resize(roi, (64, 64))
                roi_resized = roi_resized.astype('float32') / 255.0
                roi_flat = roi_resized.reshape(1, -1)

                # Hacer predicción en la ROI
                prediction = knn.predict(roi_flat)
                color = label_encoder_classes[prediction][0]

                # Mostrar el color predicho en la imagen
                cv2.putText(frame, f'Color: {color}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen con los rectángulos y colores detectados
    cv2.imshow('Color Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
