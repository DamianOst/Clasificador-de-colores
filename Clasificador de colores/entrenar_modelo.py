import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Cargar los datos procesados
imagenes = np.load("imagenes.npy")
etiquetas = np.load("etiquetas.npy")

# Aplanar las imágenes para el clasificador KNN
x_train, x_test, y_train, y_test = train_test_split(imagenes.reshape(imagenes.shape[0], -1), etiquetas, test_size=0.2, random_state=42)

# Crear modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Predecir y evaluar el modelo
y_pred = knn.predict(x_test)
print(f"Precisión: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar modelo entrenado
joblib.dump(knn, "color_clasificador_modelo.pkl")
