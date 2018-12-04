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

#Sigmoid prime
def sigmoidPrime(x):
    return (np.exp(-x)/ ((1+np.exp(-x))**2) )

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
    global weight_matrix_1,weight_matrix_2,weight_matrix_output
    init()
    input = np.array(list(d2b(0)),dtype='uint8')
    inputDec = 0
    count = 0
    while(count < 10000):
        arraystring = np.array2string(input,separator='')[1:-1]
        inputDec = b2d(arraystring)
        target = np.array(list(d2b(inputDec+1)),dtype='uint8')
        print '\n\nCount :',count
        print 'input:',input,' Dec:',inputDec

        hidden_layer_1 = sigmoid(np.dot(input,weight_matrix_1)+bais_1)
        # hidden_layer_2 = sigmoid(np.dot(hidden_layer_1,weight_matrix_2) + bais_2)
        output = sigmoid(np.dot(hidden_layer_1,weight_matrix_output)+bais_3)
        staped = step(output)
        error = np.square(target - output)*1/2

        input = input.reshape(1,4)

        # gradidant
        delta_out = np.multiply((output - target), sigmoidPrime(np.dot(hidden_layer_1,weight_matrix_output)+bais_3))
        dedw2 = np.dot(hidden_layer_1.T , delta_out)
        delta1 = np.dot(delta_out, weight_matrix_output.T) * sigmoidPrime(np.dot(input,weight_matrix_1)+bais_1)
        dedw1 = np.dot(input.T, delta1)
        # correction
        weight_matrix_1 = weight_matrix_1 - dedw1
        weight_matrix_output = weight_matrix_output - dedw2

        input = staped.reshape(-1)
        count += 1

        print 'output:',staped,' Dec:',b2d(np.array2string(staped,separator='')[2:-2])
        print 'target :',target,' Dec:',b2d(np.array2string(target,separator='')[1:-1])
        print 'error :',error
        print 'Toral error:',error.sum()

        # print "weight_matrix_1 gradidant :",dedw1
        # print "weight_matrix_1 orign :",weight_matrix_1.shape
        # print "weight_matrix_output gradidant :",dedw2
        # print "weight_matrix_output orign :",weight_matrix_output.shape

    # learned data
    print 'weight_matrix_1 :',weight_matrix_1
    print '\n weight_matrix_output :',weight_matrix_output



run()
