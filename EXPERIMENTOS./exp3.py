#!/usr/bin/env python
# coding: utf-8

# In[2]:


#para este experimento se utilizó la función de activación relu,no. de clases,ADAM
import tensorflow as tf
import keras as keras
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD,Adam


# In[3]:


learning_rate = 0.001
epochs = 20
batch_size = 120


# In[4]:


from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[5]:


X_train.shape


# In[6]:


x_trainv = X_train.reshape(60000, 784)
x_testv = X_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255


# In[7]:


print(Y_train[10000])


# In[8]:


num_classes=10
y_trainc = keras.utils.to_categorical(Y_train, num_classes)
y_testc = keras.utils.to_categorical(Y_test, num_classes)


# In[9]:


plt.figure()
plt.imshow(X_train[5])#número de imagen en el mnist
plt.colorbar()
plt.grid(False)
plt.show()


# In[10]:


#pre-procesamiento
train_images = X_train / 255.0#escalara los valores

test_images = Y_train / 255.0


# In[13]:


model = Sequential()
model.add(Dense(512, activation='relu6', input_shape=(784,)))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()


# In[15]:


#model.compile(optimizer='adam',
           #   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])


# In[16]:


model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate),metrics=['accuracy'])


# In[17]:


history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )


# In[18]:


score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)
a=model.predict(x_testv) #predicción de la red entrenada
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])


# In[ ]:




