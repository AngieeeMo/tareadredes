import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
training_data = list(training_data) 
import network 
 
net = network.Network([784, 30, 10]) 
net.Adam_momentum(training_data, 55, 10, 0.07,0.45, test_data=test_data)
