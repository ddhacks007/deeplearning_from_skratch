import numpy as np

def activation(fun, inp):
    activate_functions = {
        'sigmoid': 1/(1+ np.exp(-inp)),
    }
    return activate_functions[fun]

def derivative(fun, inp):
    derivative_fun = {
        'sigmoid': inp * (1-inp),
    }
    return derivative_fun[fun]

def create_dataset():
    input_x = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]]
    output_y = [[0], [1], [1], [0]]
    return input_x, output_y

def synopses(shape):
    print(shape)
    return np.random.normal(size = shape)

def multiply(syn0, syn1):
    return np.matmul(syn0, syn1)

def create_dataset():
    input_x = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]])
    output_y = np.array([[0], [1], [1], [0]])
    return input_x, output_y

def synopses(shape):
    print(shape)
    return np.random.normal(size = shape)

def multiply(syn0, syn1):
    return np.matmul(syn0, syn1)

def learn(learning_rate, recommended_activation):
    x, y = create_dataset()
    print(x)
    synopses_1 = synopses((3, 3))
    synopses_2 = synopses((3, 2))
    synopses_3 = synopses((2, 1))
    for i in range(1000000):
        hidden_layer_1 = activation(recommended_activation, multiply(x, synopses_1))
        hidden_layer_2 = activation(recommended_activation, multiply(hidden_layer_1, synopses_2))
        output_layer = activation(recommended_activation, multiply(hidden_layer_2, synopses_3))
        error = output_layer - y
        print('error in the ', i, ' epoch is', (sum(error)/len(x))[0] )
        delta_1 = derivative(recommended_activation, output_layer) * error 
        print(delta_1.shape)
        delta_2 = multiply(delta_1, synopses_3.T) * derivative(recommended_activation, hidden_layer_2)
        delta_3 = multiply(delta_2, synopses_2.T) * derivative(recommended_activation, hidden_layer_1)
        synopses_1 = synopses_1 - (learning_rate * multiply(x.T, delta_3)  )
        synopses_2 = synopses_2 - (learning_rate * multiply(hidden_layer_1.T, delta_2) )
        synopses_3 = synopses_3 - (learning_rate * multiply(hidden_layer_2.T, delta_1) )

learn(0.01, 'sigmoid')
