
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

def new_weights(shape,graph):
     with graph.as_default():
        weights =  tf.Variable(tf.random.truncated_normal(shape,stddev = 0.05))
     return weights   

#temp_weights = new_weights((2,3))
#temp_weights
    
def new_biases(length,graph):
    with graph.as_default():
        biases =  tf.Variable(tf.constant(0.05,shape = [length]))
    return biases

#temp_biases = new_biases(10)
#temp_biases
    
def new_conv_layer(input,             #the previos layer
                   num_input_channels, #num channel in previous layer
                   filter_size,       #height and width of filters
                   num_filters,       # num_filters
                   use_pooling = True, #use pooling layer or not
                   graph = None
                   ):
    
        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        #creatre new wieths aka filters for the given shape
        weights = new_weights(shape = shape,graph = graph)
        #create new biases one for each filter
        biases = new_biases(length = num_filters,graph = graph)
        
 
        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input = input,
                             filter = weights,
                             strides = [1,1,1,1],
                             padding = 'SAME')
        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases
        
         # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool2d(input = layer,
                                     ksize = [1,2,2,1],
                                     strides = [1,2,2,1],
                                     padding = 'SAME')
            
        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)
        
        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    
    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]
    

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.    
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer,[-1,num_features])
    
    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]
    
    return layer_flat,num_features


def new_fc_layer(input,           #the previous layer
                 num_inputs,      #num of units in prev layer
                 num_outputs,     #num of output units
                 use_relu = True,
                 graph = None): # use relu
    #createa weights and biases for the fc layer
    weights = new_weights(shape = [num_inputs,num_outputs],graph = graph)
    biases = new_biases(length = num_outputs,graph = graph)
    
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input,weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer    











