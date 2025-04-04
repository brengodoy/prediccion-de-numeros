import torch #obligatorio para usar pytorch
from torchvision import datasets # libreria de datasets donde se encuentra MNIST
from torchvision.transforms import ToTensor # libreria para transformar los datos en tensores
from torch import nn # libreria de redes neuronales
from torch.utils.data import DataLoader, random_split # DataLoader es un objeto que toma un dataset y lo divide en partes más pequeñas llamadas batches (lotes)
from modelo import RedNeuronal

# creamos el dataset 'data_mnist'
data_mnist = datasets.MNIST(
	root = "datos", # carpeta donde se almacenan los datos
	train = True, #true = 60000 imagenes, false = 10000 imagenes. Nos da las 60,000 imágenes de entrenamiento.
	download=True, # si no existen los datos, los descarga de internet
	transform=ToTensor() # transforma las imagenes a tensores. Originalmente, vienen como matrices de numpy o imágenes en formato PIL.
)

#separamos los datos en entrenamiento, validacion y prueba
torch.manual_seed(42)

# dividir el dataset en 3 datasets
train, val = random_split(data_mnist, [0.9, 0.1]) # random_split espera cantidades enteras, no porcentajes, pero internamente convierte estos valores multiplicándolos por la cantidad total de datos.
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modelo = RedNeuronal().to(device)

# entrenamos de a lotes porque 48000 imagenes es muy lento
# definir el tamaño del lote
TAM_LOTE = 1000 # 60 lotes de 1000 imagenes cada uno. 60*1000 = 60000 imagenes

train_loader = DataLoader(
	dataset=train,
	batch_size=TAM_LOTE,
	shuffle=True # mezclar datos aleatoriamente para cada lote
)

val_loader = DataLoader(
	dataset=val,
	batch_size=TAM_LOTE,
	shuffle=False
)

#entrenar y validar el modelo
#hiperparametros
TAZA_APRENDIZAJE = 0.1 # suele estar entre 0.001 y 0.1
EPOCHS = 20 # num de iteraciones de entrenamiento

#funcion de perdida y optimizador
fn_perdida = nn.CrossEntropyLoss() # funcion ideal cuando tenemos varias clases, en este caso hay 10 clases. Ya incluye Softmax
optimizador = torch.optim.SGD(modelo.parameters(), lr=TAZA_APRENDIZAJE) # SGD fácil de entender y funciona bien en problemas pequeños.

def train_loop(dataloader, model, loss_fn, optimizer):
    # cantidad de datos de entrenamiento y cantidad de lotes
    train_size = len(dataloader.dataset) # longitud de 60000 imagenes
    nlotes = len(dataloader) # Depende del batch_size que definimos. batch_size = 1000, entonces hay 60,000 / 1000 = 60 lotes. Pensalo como una pila de 60,000 cartas que querés repartir en sobres con 1,000 cartas por sobre. Al final, te quedan 60 sobres
    
    #indicarle a pytorch que entrenamos el modelo
    model.train()
    
    #inicializar acumuladores de perdida y exactitud
    perdida_train, exactitud = 0, 0
    
    # enumerate(dataloader) no solo nos da cada lote, sino que también nos da su índice (nlote).
    # nlote es el indice dentro de dataloader (de 0 a 59)
    # X son las imágenes del lote actual
    # y son las etiquetas reales del lote actual
    for nlote, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # asegurarnos de que los datos y el modelo estén en el mismo dispositivo.
        
        #forward propagation
        logits = model(X) # logits son las predicciones, se ejecuta la funcion 'forward' dentro de RedNeuronal (y muchas otras cosas que no vemos)
        
        #backpropagation
        loss = loss_fn(logits, y)  # 1️ calcular la pérdida
        loss.backward()            # 2️ Hacer el cálculo de gradientes
        optimizer.step()           # 3️ Actualizar los pesos usando los gradientes
        optimizer.zero_grad()      # 4️ Resetear los gradientes para la próxima iteración
        
        # acumular valores de perdida y de exactitud
        perdida_train += loss.item() # .item() convierte un tensor de un solo valor en un float.
        exactitud += (logits.argmax(1) == y).type(torch.float).sum().item() # Calcula cuántas predicciones fueron correctas en el batch
        # 1. logits.argmax(1) -> obtiene la clase con mayor probabilidad por muestra
        # 2. == y -> compara con la etiqueta real, devuelve True/False
        # 3. type(torch.float) -> convierte True/False en 1.0/0.0
        # 4. sum().item() -> cuenta los aciertos y lo convierte en número Python
        
        # imprimir la evolucion del entrenamiento (cada 10 lotes)
        if nlote % 10 == 0:
            # obtener el valor de la perdida y el num de datos procesados
            ndatos = nlote * TAM_LOTE
            print("\tPerdida: " + str(loss.item()) + " [" + str(ndatos) + "/" + str(train_size) + "]")
            
    # calcular el promedio de perdida y exactitud
    perdida_train /= nlotes # perdida_train = perdida_train / nlotes. La pérdida se promedia por lote porque cada lote tiene una sola pérdida.
    exactitud /= train_size # exactitud = exactitud / train_size. La exactitud se promedia sobre todos los datos porque contamos imágenes correctas, no un solo valor por lote.
    print(f"Entrenamiento: Perdida promedio: {perdida_train:>8f} Exactitud: {(100*exactitud):>8.2f}%")
    
def val_loop(dataloader,model,loss_fn):
    # cantidad de datos de validacion y cantidad de lotes
    val_size = len(dataloader.dataset) # longitud de 60000 imagenes
    nlotes = len(dataloader) # Depende del batch_size que definimos. batch_size = 1000, entonces hay 60,000 / 1000 = 60 lotes. Pensalo como una pila de 60,000 cartas que querés repartir en sobres con 1,000 cartas por sobre. Al final, te quedan 60 sobres
    
    # indicarle a pytorch que validaremos el modelo
    model.eval()
    
    # inicializar acumuladores de perdida y exactitud
    perdida_val, exactitud = 0, 0
    
    with torch.no_grad(): # no calcules los gradientes, porque no estoy entrenando el modelo aqui
        for X, y in dataloader:
            # mover X e y a la gpu
            X, y = X.to(device), y.to(device)
            
            #propagacion hacia adelante (predicciones)
            logits = model(X)
            
            #acumular valores de perdida y exactitud
            perdida_val += loss_fn(logits, y).item()
            exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()
    
    #calcular perdida promedio y exactitud promedio
    perdida_val /= nlotes
    exactitud /= val_size
    
    print(f"Validacion: Perdida promedio: {perdida_val:>8f} Exactitud: {(100*exactitud):>8.2f}%")
    
for t in range(EPOCHS):
    print("Iteracion:" + str(t+1))
    train_loop(train_loader,modelo,fn_perdida,optimizador)
    val_loop(val_loader,modelo,fn_perdida)

# Después de entrenar el modelo, lo guardo
torch.save(modelo.state_dict(), 'modelo.pth')
print("Entrenamiento terminado!")