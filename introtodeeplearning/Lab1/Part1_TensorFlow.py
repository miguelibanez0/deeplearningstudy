import tensorflow as tf

# Download and import the MIT Introduction to Deep Learning package
# !pip install mitdeeplearning --quiet
import mitdeeplearning as mdl

import numpy as np
import matplotlib.pyplot as plt

# El rango (rank) de un tensor proporciona el número de dimensiones (n-dimensiones); también puedes considerarlo como el orden o grado del tensor.
# La forma (shape) de un tensor define su número de dimensiones y el tamaño de cada dimensión.

# Veamos primero los tensores 0-d, de los cuales un escalar es un ejemplo:

sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)
print('\n \n')
print ('---------- EJEMPLO DE TENSORES -------------------')
print('\n')

print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

# `sport` is a 0-d Tensor  --> Rank 0 dimensiones correspondiente a un escalar.
# `number` is a 0-d Tensor --> Rank 0 dimensiones correspondiente a un escalar.

# Se pueden utilizar vectores y listas para crear tensores unidimensionales:

sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))

# `sports` is a 1-d Tensor with shape: [2]   ---> Rank 1 dimension, Shape 2 tamaño de la dimensión, correspondiente a ["Tennis", "Basketball"].

#####    `sports` is a 1-d Tensor with shape: [2] ####
#                 | Tennis  Basketball |             #
######################################################

print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))


# `numbers` is a 1-d Tensor with shape: [3]  ---> Rank 1 dimension, Shape 3 tamaño de la dimensión, correspondiente a [3.141592, 1.414213, 2.71821]  .

#####    `sports` is a 1-d Tensor with shape: [3] ####
#          | 3.141592 1.414213 2.71821 |             #
######################################################


### Defining higher-order Tensors ###

##TOD0 Define a 2-d Tensor

matrix = tf.constant([[5, 10, 15, 20],[25, 30, 35, 40],[45, 50, 55, 60]], tf.int64)

# Matrix 2-d Tensor #
#  | 5  10 15 20 |  #
#  | 25 30 35 40 |  #
#  | 45 50 55 60 |  #
#####################

print("`matrix` is a {}-d Tensor with shape: {}".format(tf.rank(matrix).numpy(), tf.shape(matrix)))

assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(matrix).numpy() == 2

##TOD0 Define a 4-d Tensor
# Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
#   You can think of this as 10 images where each image is RGB 256 x 256.

images = tf.zeros([10, 256, 256, 3], tf.int32)
print("`images` is a {}-d Tensor with shape: {}".format(tf.rank(images).numpy(), tf.shape(images)))

assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"


row_vector = matrix[1]
column_vector = matrix[:,1]
scalar = matrix[0, 1]

# Matrix 2-d Tensor #
#  | 5  10 15 20 |  #
#  | 25 30 35 40 |  #
#  | 45 50 55 60 |  #
#####################

print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))

print('\n \n')
print ('---------- EJEMPLO DE CALCULOS  -------------------')
print('\n')

# Create the nodes in the graph, and initialize values
a = tf.constant(15)
b = tf.constant(61)

# Add them!
c1 = tf.add(a,b)
c2 = a + b # TensorFlow overrides the "+" operation so that it is able to act on Tensors
print(c1)
print(c2)

### Defining Tensor computations ###

# Construct a simple computation function
def func(a,b):
   
  #TOD0: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply).

  c = tf.add(a,b)
  d = tf.subtract(b,1)
  e = tf.multiply(c,d)
  return e

# Now, we can call this function to execute the computation graph given some inputs a,b:

# Consider example values for a,b
a, b = 1.5, 2.5
# Execute the computation
e_out = func(a,b)
print(e_out)

matrix1 = tf.constant([[5, 10]], tf.int64)
a1 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b1 = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
    
print("`matrix1` is a {}-d Tensor with shape: {}".format(tf.rank(matrix1).numpy(), tf.shape(matrix1)))
print("`a1` is a {}-d Tensor with shape: {}".format(tf.rank(a1).numpy(), tf.shape(matrix)))
print("`b1` is a {}-d Tensor with shape: {}".format(tf.rank(b1).numpy(), tf.shape(matrix)))


print('\n \n')
print ('---------- EJEMPLO DE CAPA CON 2 Variables y cálculo y = w . x + b. -------------------')
print('\n')
### Here's a basic example: a layer with two variables, w and b, that returns y = w . x + b. 
# It shows how to implement build() and call(). Variables set as attributes of a layer are 
# tracked as weights of the layers (in layer.weights). ###

class SimpleDense(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    # Create the state of the layer (weights)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    # Defines the computation
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias
        
        # return tf.subtract(inputs,0)
        # tf.Tensor(
        # [[1. 1.]
        #[1. 1.]], shape=(2, 2), dtype=float32)
        
        # return tf.subtract(self.kernel,0)
        # tf.Tensor(
        # [[ 0.6193409   0.40583587  0.08732152 -0.10335231]
        #[-0.8345103   0.5195904  -0.24846578 -0.14063239]], shape=(2, 4), dtype=float32)
        
        # return tf.subtract(self.bias,0) 
        # tf.Tensor([0. 0. 0. 0.], shape=(4,), dtype=float32)

# Instantiates the layer.
linear_layer = SimpleDense(4)
# print(linear_layer)
# print('\n')
# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
assert len(linear_layer.weights) == 2

# These weights are trainable, so they're listed in `trainable_weights`:
assert len(linear_layer.trainable_weights) == 2

print('Tensor with all elements set to one (1).')
print(tf.ones((2, 2))) # Creates a tensor with all elements set to one (1).
print('\n')
print('Tensor resultado del calculo y = w . x + b.')
print(y)
# print('\n')
# print(y.numpy())

print('\n \n')
print ('----------Defining a network Layer -------------------')
print('\n')

### Defining a network Layer ###

# n_output_nodes: number of output nodes
# input_shape: shape of the input
# x: input to the layer

class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    # Create the state of the layer (weights)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="weight",
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    # Defines the computation
    def call(self, inputs):
        #TOD0: define the operation for z (hint: use tf.matmul that Multiplies matrix a by matrix b, producing a * b.) 
        z = tf.matmul(inputs, self.W) + self.b
        #TOD0: define the operation for out (hint: use tf.sigmoid)
        return tf.sigmoid(z)

# Since layer parameters are initialized randomly, we will set a random seed for reproducibility
tf.keras.utils.set_random_seed(1)
# Instantiates the layer.
layer = OurDenseLayer(3)
layer.build((1,2))
# y1 = layer(tf.constant(((1, 2.)), shape=(1,2)))
x_input = tf.constant(((1, 2.)), shape=(1,2))
y1 = layer.call(x_input)

# test the output result of computation y1 = Sigma(W . x + b).!
print('Tensor resultado del calculo y = Sigma(W . x + b).')
print(y1.numpy())
# mdl.lab1.test_custom_dense_layer_output(y1) Reviasar funcionamiento de código.
