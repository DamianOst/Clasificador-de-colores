from PIL import Image
import os
import random

# Crear una carpeta para guardar las imágenes generadas
os.makedirs('datasets/rojo', exist_ok=True)
os.makedirs('datasets/azul', exist_ok=True)
os.makedirs('datasets/verde', exist_ok=True)

def create_color_image(color, size=(100, 100), filename="imagen.jpg"):
    img = Image.new("RGB", size, color)
    img.save(filename)
    print(f'{filename} creada')

# Generar múltiples imágenes de diferentes tamaños y variaciones de color
for i in range(50):  # Generar 50 imágenes para cada color
    # Variación aleatoria en el color
    red_variation = (255, random.randint(0, 50), random.randint(0, 50))
    blue_variation = (random.randint(0, 50), random.randint(0, 50), 255)
    green_variation = (random.randint(0, 50), 255, random.randint(0, 50))
    
    # Variación aleatoria en el tamaño
    size = (random.randint(50, 150), random.randint(50, 150))
    
    # Crear imágenes
    create_color_image(red_variation, size, filename=f"datasets/rojo/rojo_{i}.jpg")
    create_color_image(blue_variation, size, filename=f"datasets/azul/azul_{i}.jpg")
    create_color_image(green_variation, size, filename=f"datasets/verde/verde_{i}.jpg")
