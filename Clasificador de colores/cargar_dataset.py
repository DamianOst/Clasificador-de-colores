import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def cargar_imagenes_del_folder(folder):
    imagenes = []
    etiquetas = []
    for etiqueta in os.listdir(folder): 
        ruta_imagen = os.path.join(folder, etiqueta) 
        for nombre_archivo in os.listdir(ruta_imagen):
            imagen = cv2.imread(os.path.join(ruta_imagen,nombre_archivo))
            if imagen is not None:
                imagen = cv2.resize(imagen,(64,64)) # Redimensiona el tamaño
                imagenes.append(imagen)
                etiquetas.append(etiqueta) # La etiqueta sera el nombre de la carpeta
    return np.array(imagenes), np.array(etiquetas) 

def preproceso_de_datos(imagenes, etiquetas):
    # Escalar los valores de pixeles 0 y 1
    imagenes = imagenes.astype('float32')/255.0
    # Codificar las etiquetas
    le = LabelEncoder()
    etiquetas = le.fit_transform(etiquetas)
    return imagenes, etiquetas, le

imagenes, etiquetas = cargar_imagenes_del_folder('datasets')
imagenes, etiquetas, le = preproceso_de_datos(imagenes, etiquetas)
np.save('imagenes.npy',imagenes)
np.save('etiquetas.npy', etiquetas)
np.save('etiqueta_encoder.npy',le.classes_)