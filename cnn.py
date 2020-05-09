import math
import os
import random
import time
import warnings
from datetime import timedelta

import cv2
import LayersConstructor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Preprocessor
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.classification import accuracy_score
from tensorflow.python.framework import ops

#convolution layer1
filter_size1 = 3
num_filters1 = 32

#convolution layer2
filter_size2 = 3  
num_filters2 = 32

#convolution layer3
filter_size3 = 3  
num_filters3 = 64

#fully connvected layer
fc_size = 128

#image properties
image_size = 128
num_channels = 3
image_size_flat = image_size * image_size * num_channels
image_shape = (image_size,image_size)

#classes_info
classes = ['dogs','cats']
num_classes = len(classes)

#cnn hyperparameters
learning_rate = 1e-4
batch_size = 32
validation_size = 0.16
early_stopping = False

checkpoint_dir = "models/"

#loading data
train_path = 'dataset/train/'
test_path = 'dataset/test'

data = Preprocessor.read_train_set(train_path,image_size,classes,validation_size)
test_images, test_ids = Preprocessor.read_test_set(test_path, image_size)

print("Size of:")
print("  - Training-set:\t\t{}".format(len(data.train.labels)))
print("  - Test-set:\t\t{}".format(len(test_images)))
print("  - Validation-set:\t{}".format(len(data.valid.labels)))


def plot_images(images,cls_true,cls_pred = None,index = 0):
    

    #create figure with 3*3 subplots
    plt.figure(figsize = (9,9))
    for i in range(9):
        plt.subplot(3,3,i+1,xticks = [],yticks = [])
        plt.imshow(images[index+i])
        plt.title(str(cls_true[index+i]))
    plt.show()
        

plot_images(data.train.images,data.train.labels,cls_pred = None,index = 9)

#making placeholders for input and output
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='x')
x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
print(x_image.graph)
#for the first layer the input image will have the shape -[None,128,128,3]

layer_conv1,weights_conv1 = LayersConstructor.new_conv_layer(input = x_image,num_input_channels = num_channels,
                                                            filter_size = filter_size1,num_filters = num_filters1,
                                                            use_pooling = True,graph = tf.get_default_graph())

#for the first layer the output image will have the shape - [None,64,64,32], shaoe of weights is (3,3,3,32)

layer_conv2,weights_conv2 = LayersConstructor.new_conv_layer(input = layer_conv1,num_input_channels = num_filters1,
                                                             filter_size = filter_size2,num_filters = num_filters2,
                                                             use_pooling = True,graph = tf.get_default_graph())

#for the second layer, the output wil have shape - [None,32,32,32], weights - [3,3,32,32]

layer_conv3,weights_conv3 = LayersConstructor.new_conv_layer(input = layer_conv2,num_input_channels = num_filters2,
                                                             filter_size = filter_size3,num_filters = num_filters3,
                                                             use_pooling = True,graph = tf.get_default_graph())


#the third layer will have an output of [None,16,16,64], weights - [3,3,32,64]

layer_flat,num_features = LayersConstructor.flatten_layer(layer_conv3)

#flatten layer will have an output of [None,16384]

#the number of features input to this will be num_features returened by the last layer. 32*32*64

layer_fc1 = LayersConstructor.new_fc_layer(input = layer_flat,num_inputs = num_features,
                                           num_outputs = fc_size,use_relu = True,graph = tf.get_default_graph())
#output for fc_1 will be [None,128]

layer_fc2 = LayersConstructor.new_fc_layer(input = layer_fc1,num_inputs = fc_size,
                                           num_outputs = num_classes,use_relu = False,graph = tf.get_default_graph())
#final layer will output as [None,2]

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred,axis = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer_fc2,labels = y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init_op = tf.global_variables_initializer()

train_batch_size = batch_size

acc_list = []
val_acc_list = []


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,session):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    return acc, val_acc
  
# Counter for total number of iterations performed so far.
total_iterations = 0
iter_list = []

def optimize(num_iterations):
    
    #make sure that we use the global variable for total_iterations
    global total_iterations,g
    
    with tf.Session() as sess:
        sess.run(init_op)    
    
        #start time used for printing time-usage
        start_time = time.time()
        
        best_value_loss = float("inf")
        patience = 0
        
        for i in range(total_iterations,total_iterations + num_iterations):
            
            if(i%100 == 0):
                print('iterationm number is ',i)
            
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
            
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, flattened image shape]
            
            x_batch = x_batch.reshape(batch_size,image_size_flat)
            x_valid_batch = x_valid_batch.reshape(batch_size,image_size_flat)
            
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            feed_dict_train = {x:x_batch,y_true:y_true_batch}
            feed_dict_validate = {x:x_valid_batch,y_true:y_valid_batch}
            
            
            sess.run(optimizer,feed_dict_train)
            
            # Print status at end of each epoch (defined as full pass through training Preprocessor).
            if i % int(data.train.num_examples/batch_size) == 0: 
                val_loss = sess.run(cost, feed_dict=feed_dict_validate)
                epoch = int(i / int(data.train.num_examples/batch_size))
                
                acc, val_acc = print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,session = sess)
                msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
                print(msg.format(epoch + 1, acc, val_acc, val_loss))
                print(acc)
                acc_list.append(acc)
                val_acc_list.append(val_acc)
                iter_list.append(epoch+1)
#                
#                if early_stopping:    
#                    if val_loss < best_val_loss:
#                        best_val_loss = val_loss
#                        patience = 0
#                    else:
#                        patience += 1
#                    if patience == early_stopping:
#                        break
                    
            # Update the total number of iterations performed.
            total_iterations += num_iterations
    
            # Ending time.
            end_time = time.time()
    
            # Difference between start and end-times.
            time_dif = end_time - start_time
        
            # Print the time-usage.
            print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


#Evaluation and optimization 
optimize(num_iterations=10000)
print(acc_list)
# Plot loss over time
plt.plot(iter_list, acc_list, 'r--', label='CNN training accuracy per iteration', linewidth=4)
plt.title('CNN training accuracy per iteration')
plt.xlabel('Iteration')
plt.ylabel('CNN training accuracy')
plt.legend(loc='upper right')
plt.show()

def plot_example_errors(cls_pred, correct):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.valid.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    
    # Compute the precision, recall and f1 score of the classification
    p, r, f, s = precision_recall_fscore_support(cls_true, cls_pred, average='weighted')
    print('Precision:', p)
    print('Recall:', r)
    print('F1-score:', f)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def print_validation_accuracy(show_example_errors=False, show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.valid.images[i:j, :].reshape(batch_size, image_size_flat)
        

        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images, y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)       
       
        

# Plot loss over time
plt.plot(iter_list, val_acc_list, 'r--', label='CNN validation accuracy per iteration', linewidth=4)
plt.title('CNN validation accuracy per iteration')
plt.xlabel('Iteration')
plt.ylabel('CNN validation accuracy')
plt.legend(loc='upper right')
plt.show()  

print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)
plt.axis('off')        














