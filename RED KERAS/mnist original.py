#!/usr/bin/env python
# coding: utf-8

# In[46]:


import tensorflow as tf
import keras as keras
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD


# In[20]:


learning_rate = 0.001
epochs = 30
batch_size = 120


# In[11]:


from tensorflow.keras.datasets import mnist#cargar los datos desde internet
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[12]:


X_train.shape#dimensión de los datos


# In[22]:


x_trainv = X_train.reshape(60000, 784)#redimensionar la matriz de datos
x_testv = X_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')#tipo de dato de salida para que no se vaya a cerlo

x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255


# In[24]:


print(Y_train[10000])


# In[25]:


num_classes=10
y_trainc = keras.utils.to_categorical(Y_train, num_classes)#devuelve una matriz de valores binarios no. de filas igual a la longitud del vector de entrada y un número de columnas igual al número de clases.
y_testc = keras.utils.to_categorical(Y_test, num_classes)


# In[14]:


plt.figure()
plt.imshow(X_train[2])#número de imagen en el mnist
plt.colorbar()
plt.grid(False)
plt.show()


# In[15]:


#otra forma de pre-procesamiento
train_images = X_train / 255.0#escalara los valores

test_images = Y_train / 255.0


# In[37]:


model = Sequential()##modelo
model.add(Dense(512, activation='sigmoid', input_shape=(784,)))##capa de entrada
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()


# In[38]:


#model.compile(optimizer='adam',
           #   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])


# In[47]:


model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate),metrics=['accuracy'])
##funcion de perdida,optimizador,taza de aprendizaje,métrica


# In[48]:


history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )


# In[49]:


score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=model.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])


# In[ ]:




