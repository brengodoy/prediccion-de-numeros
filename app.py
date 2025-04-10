from tkinter import *
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
import torch
from model import NeuralNetwork  # Solo importar la clase NeuralNetwork

def delete_all():
    for item in drawing_area.find_all():
        drawing_area.delete(item)
    result_label.config(text=f"Predicción: ")

def start_drawing(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    drawing_area.create_line((last_x, last_y, x, y), fill="black", width=10, capstyle=ROUND, smooth=TRUE)
    last_x, last_y = x, y
    save()

def save():
    file_name = "imagen.png"
    
    # Crear una imagen en blanco con el mismo tamaño que el canvas
    image = Image.new("RGB", (200, 200), "white")
    draw_image = ImageDraw.Draw(image)

    # Recorrer todos los items en el canvas y dibujarlos sobre la imagen
    for item in drawing_area.find_all():
        coords = drawing_area.coords(item)
        if len(coords) == 4:
            draw_image.line([coords[0], coords[1], coords[2], coords[3]], fill="black", width=10)

    # Convertir la imagen a escala de grises
    image = image.convert("L")

    # Invertir la imagen (fondo blanco, números en negro)
    image = ImageOps.invert(image)

    # Redimensionar la imagen a 28x28 píxeles
    image = image.resize((28, 28))

    # Guardar la imagen
    image.save(file_name)
    
    # Transformar la imagen a tensor (sin necesidad de invertir ni cambiar tamaño)
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0)  # Añadir una dimensión extra para representar el batch (1 imagen)

    # 1. Definir la misma estructura del modelo
    model = NeuralNetwork()  # Usamos la misma estructura del modelo

    # 2. Cargar los pesos del modelo entrenado
    model.load_state_dict(torch.load('model_number.pth', weights_only=True))  # Asegúrate de que el archivo 'model.pth' esté en la misma carpeta
    model.eval()  # Cambiar el modelo a modo de evaluación
    
    # Hacer la predicción
    with torch.no_grad():  # No calcular gradientes
        logits = model(img_tensor)  # Hacer la predicción
        y_pred = logits.argmax(1).item()  # Obtener la clase con la mayor probabilidad

    result_label.config(text=f"Predicción: {y_pred}")

# Crear ventana
canvas = Tk()
canvas.title("Dibuje un número")
result_label = Label(canvas, text="Predicción: ", font=("Arial", 16))
result_label.pack()

# Crear área de dibujo
drawing_area = Canvas(canvas, width=200, height=200, bg="white")
drawing_area.pack(expand=YES, fill=BOTH)

# Conectar eventos del mouse para dibujar
drawing_area.bind("<Button-1>", start_drawing)  
drawing_area.bind("<B1-Motion>", draw)          

# boton para borrar
delete_btn = Button(canvas, text="Borrar", bg="red", fg="white", command=delete_all)
delete_btn.pack()

# Ejecutar loop principal
canvas.mainloop()