# importe a biblioteca opencv
import cv2
import numpy as np
import tensorflow as tf
  
model = tf.keras.models.load_model("C:/Users/hlaic/Documents/Projetos/Python/aula110/keras_model.h5")
# defina um objeto de captura de vídeo
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture o vídeo quadro a quadro
    ret, frame = vid.read()
    
    img = cv2.resize(frame,(224,224))
    test_img = np.array(img,dtype=np.float32)
    test_img = np.expand_dims(test_img,axis=0)
    nimg = test_img/255.0
    prediction = model.predict(nimg)
    print(prediction)
  
    # Exiba o quadro resultante
    cv2.imshow('quadro', frame)
      
    # Saia da tela com a barra de espaço
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# Após o loop, libere o objeto capturado
vid.release()

# Destrua todas as janelas
cv2.destroyAllWindows()