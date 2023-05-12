
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
    
def new_biases(length,graph):
    with graph.as_default():
        biases =  tf.Variable(tf.constant(0.05,shape = [length]))
    return biases

def new_conv_layer(input,             
                   num_input_channels,
                   filter_size,      
                   num_filters,      
                   use_pooling = True, 
                   graph = None
                   ):
   
        shape = [filter_size, filter_size, num_input_channels, num_filters]
      
        weights = new_weights(shape = shape,graph = graph)
        
        biases = new_biases(length = num_filters,graph = graph)
       
        layer = tf.nn.conv2d(input = input,
                             filter = weights,
                             strides = [1,1,1,1],
                             padding = 'SAME')
       
        layer += biases
        
        
        if use_pooling:
            
            layer = tf.nn.max_pool2d(input = layer,
                                     ksize = [1,2,2,1],
                                     strides = [1,2,2,1],
                                     padding = 'SAME')
            
       
        layer = tf.nn.relu(layer)
        
       
        return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    
    
    num_features = layer_shape[1:4].num_elements()
    
    
    layer_flat = tf.reshape(layer,[-1,num_features])
    
   
    
    return layer_flat,num_features


def new_fc_layer(input,         
                 num_inputs,     
                 num_outputs,     
                 use_relu = True,
                 graph = None): 
    weights = new_weights(shape = [num_inputs,num_outputs],graph = graph)
    biases = new_biases(length = num_outputs,graph = graph)
    
    
    layer = tf.matmul(input,weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer    











