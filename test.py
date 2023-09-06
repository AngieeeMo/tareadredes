import mnist_loader ## importar código/bloque de los datos

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
training_data = list(training_data) 
import network #importar código de la red
 
net = network.Network([784, 30, 10]) ##parametros para entrenar
net.SGD(training_data, 55, 10, 0.07, test_data=test_data)
