import numpy as np
import time

#Hyper parameter
no_layers = 3
no_input_node = 4
no_hidden_node_1 = 6
no_hidden_node_2 = 6
no_output_node = 4

weight_matrix_1 = 0
weight_matrix_2 = 0
weight_matrix_output = 0
bais_1 = 0
bais_2 = 0
bais_3 = 0

#initialize matixs
def init():
    global weight_matrix_1,weight_matrix_2,weight_matrix_output,bais_1,bais_2,bais_3
    weight_matrix_1 = np.random.uniform(-1,1,(no_input_node,no_hidden_node_1))
    weight_matrix_2 = np.random.uniform(-1,1,(no_hidden_node_1,no_hidden_node_2))
    weight_matrix_output = np.random.uniform(-1,1,(no_hidden_node_2,no_output_node))
    bais_1 = np.random.uniform(-1,1,(1,no_hidden_node_1))
    bais_2 = np.random.uniform(-1,1,(1,no_hidden_node_2))
    bais_3 = np.random.uniform(-1,1,(1,no_output_node))


# activation * Sigmoid
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Step Activation
def step(x):
    return (np.where(x > 0.5,1,0))

# bninary To desimal
def b2d(binary):
    return int(binary,2)

# desimal to binary
def d2b(decimal):
    return format(decimal,'06b')[2:]


# Run
def run():
    init()
    input = np.array(list(d2b(0)),dtype='uint8')
    inputDec = 0
    count = 0
    while(count < 1):
        inputDec = b2d(np.array2string(input,separator='')[1:-1])
        target = np.array(list(d2b(inputDec+1)),dtype='uint8')
        print '\n\nCount :',count
        print 'input:',input,' Dec:',inputDec

        hidden_layer_1 = sigmoid(np.dot(input,weight_matrix_1)+bais_1)
        # hidden_layer_2 = sigmoid(np.dot(hidden_layer_1,weight_matrix_2) + bais_2)
        output = sigmoid(np.dot(hidden_layer_1,weight_matrix_output)+bais_3)
        steped = step(output)
        error = np.square(target - output)*1/2
        input = steped
        count += 1
        print 'output:',steped,' Dec:',b2d(np.array2string(steped,separator='')[2:-2])
        print 'target :',target,' Dec:',b2d(np.array2string(target,separator='')[1:-1])
        print 'error :',error
        print 'Toral error:',error.sum()

run()
