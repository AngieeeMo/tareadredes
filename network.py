# %load network.py



#### Libraries
# Standard library
import random # está libreria nos permite obtener datos aleatorios para alimentar 

# Third-party libraries
import numpy as np

class Network(object):##definimos la clase que será la neurona

    def __init__(self, sizes):##es la función con los parametros no de neuronas por capas y pesos

        self.num_layers = len(sizes)#tamaño de capas
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)#esta funcion crea los pesos aleatoriamente
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):#valor de inicializacion por función sigmoide
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)##producto punto y le sumamos la bias
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,##obtiene los pesos para el backpropagation y evita que se atore
            test_data=None):


        training_data = list(training_data)##lista de datos para entrenar
        n = len(training_data)#nos da el tamaño de training data

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)##reorganizamos aleatoriamente los datos
            mini_batches = [
                training_data[k:k+mini_batch_size]#reparte los datos entre el númeroi de minibatches
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)##el.update le actualiza al minibatch
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))#este ciclo nos permite visualizar 
                #datos o epocas
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):#la función que actualiza el minibatch
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]#llena de ceros
        #las listas que definimos se van llenando de la suma del grad por minibatch
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)#funcion de coste
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]#A los nuevos datos que obtuvimos del 
        #grad del minibatch  se les resta el valor del aprendizaje
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # lista de todas las activaciones de capa en capa
        zs = [] # definimos una lista a la que le vamos a meter los pesos por capa
        for b, w in zip(self.biases, self.weights):#ciclo para  ir juntando pesos y bias por capa
            z = np.dot(w, activation)+b
            zs.append(z)#añade
            activation = sigmoid(z)#aplica la función sigmoide a mi producto punto(hadamard) antes definido
            activations.append(activation)#añade a activations cada valor previamente calculado
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])#calcula el error de atrás hacía adelante es decir 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in range(2, self.num_layers):#errores para el gradiente 
            z = zs[-l]
            sp = sigmoid_prime(z)#derivada
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)#función para argumento mayor de la función feedforward para calcular 
                        #la salida de la red
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)#regersa la suma

    def cost_derivative(self, output_activations, y):##funcion de costo en relacion con las salidas
      
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))##función sigmoide

def sigmoid_prime(z):##derivada de la función sigmoide
    return sigmoid(z)*(1-sigmoid(z))
