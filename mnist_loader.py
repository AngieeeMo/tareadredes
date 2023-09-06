# %load mnist_loader.py
##todo lo fui documentando primero en un notebook porque no quería correr en mi lap
import pickle ##Esta libreria nos permite serializar objetos es decir convertir un objeto en base a 
#un lenguaje específico y deserializar que es el proceso análogo de string a objeto)


import gzip ##para comprimir y descomprimir archivo.zip como el mnist.pkl.gz
# Third-party libraries
import numpy as np ## nos permite realizar los cálculos con arrays
def load_data(): ##vamos a definir una función     
 f = gzip.open('mnist.pkl.gz', 'rb') #abre el archivo.zip, el modo rb, predeterminado como lectura de datos binarios) 
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")#definimos las variables a las que con
    #.load vamos a deserealizar y a dodificar en ascii global
    f.close()# cierra el mnist.pk
    return (training_data, validation_data, test_data)
#regresa las variables en datos de entrenamiento, datos de validación, y de prueba 

def load_data_wrapper():## pre procesamiento de los datos

    tr_d, va_d, te_d = load_data()##a estas variables les aplicamos la función
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] ##reformamos una matriz de 784 x1 para la lista tr_d
    #comenzando por el elemento 0
    training_results = [vectorized_result(y) for y in tr_d[1]]##vamos a definir antes la función 
    training_data = zip(training_inputs, training_results)##como un ziper de una chamrra retorna un nuevo iterable 
    #cuyos elementos son tuplas que contienen un elemento de training data con otro de training results
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])#proceso análogo al zip para training data
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)#regresa los nuervos arrayas que formamos en la función.

def vectorized_result(j):##esta función toma un entero 
    e = np.zeros((10, 1))#matriz de ceros de 10x1
    e[j] = 1.0
    return e
