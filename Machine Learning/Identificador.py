from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import cv2
import numpy as np
import os

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r", encoding="utf-8").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def menu():
    print("\n[Identificador de Pragas]\n.\n.\n.")
    entry = int(input("[1] - Webcam\n[2] - Arquivos\n[0] - Sair\n"))
    if entry == 1:
        cam()
    elif entry == 2:
        file()
    elif entry == 0:
        exit(0)
    else:
        print("Entrada Inválida")

from PIL import Image

def file():
    # Replace this with the path to your image
    name = input("Digite o nome da Imagem selecionada: ")
    image_path = seek(name)
    if image_path:
        # Open the image
        with Image.open(image_path) as image:
            # Convert the image to RGBA
            image = image.convert("RGBA")
            
            # Remove the alpha channel
            image = image.convert("RGB")
            
            # Resize and crop the image
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # Convert the image to numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", confidence_score)
    else:
        print("\nImagem não encontrada.")

def seek(name):
            path_jpg = "Assets\{}.jpg".format(name)
            path_jpeg = "Assets\{}.jpeg".format(name)
            path_png = "Assets\{}.png".format(name)

            if os.path.exists(path_jpeg):
                return path_jpeg
            elif os.path.exists(path_jpg):
                return path_jpg
            elif os.path.exists(path_png):
                return path_png
            else:
                return None

def cam():
    camera = cv2.VideoCapture(0)
    while True:
        success,img = camera.read()
        imgS = cv2.resize(img, (224, 224))
        image_array = np.asarray(imgS)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        indexVal = np.argmax(prediction)

        cv2.putText(img, str(class_names[indexVal]),(50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
        print(class_names[indexVal])

        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair do loop
            break

    camera.release()  # Liberar a câmera após sair do loop
    cv2.destroyAllWindows()

while(True):
    menu()