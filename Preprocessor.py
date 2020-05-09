import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_train(train_path,image_size,classes):
    
    images = []
    labels = []
    ids = []
    cls = []
    
    print('reading training images')
    for fld in classes:
        tick = 0
        idx = classes.index(fld)
        print('loading {} files, index - {}'.format(fld,idx))
        path = os.path.join(train_path,fld,'*g')
        #print(path)
        files = glob.glob(path)
#        print(files)
        for f1 in files:
            tick+= 1
            if(tick % 500 == 0):
                print(tick)
            if tick == 3000:
                break
            image = cv2.imread(f1)
            image = cv2.resize(image,(image_size,image_size),interpolation = cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[idx] = 1
            labels.append(label)
            flbase = os.path.basename(f1)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images) 
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    
#    print(images.shape)
#    print(labels.shape)
#    print(ids.shape)
#    print(cls.shape)
#    
    return images,labels,ids,cls

#
#train_path = "dataset/training_data/"        
#train_images,train_labels,train_ids,train_cls = load_train(train_path,64,['cats','dogs']) 

def load_test(test_path,image_size):
    path = os.path.join(test_path, '*g')
    files = glob.glob(path)
    
    X_test = []
    X_test_id = []
    print('Reading test images')
    print('the len of files is',len(files))
    tick = 0
    for f1 in files:
        tick += 1
        if(tick%500 == 0):
            print(tick)
        if(tick == 3000):
            break
        image = cv2.imread(f1)
        image = cv2.resize(image,(image_size,image_size),interpolation = cv2.INTER_LINEAR)
        X_test.append(image)
        flbase = os.path.basename(path)
        X_test_id.append(flbase)
        
    #because we are not creating a class for test data, we will do normalisation here
    X_test = np.array(X_test,dtype = np.uint8)
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    return X_test,X_test_id
    
    
#test_path = "dataset/test"
#X_test,X_test_id = load_test(test_path,64)
#print(X_test.shape)
#plt.imshow(X_test[0])
#       
#    

class Dataset(object):
    
    def __init__(self,images,labels,ids,cls):
        self._num_examples = images.shape[0]
        
        #normalizing the images
        images = images.astype(np.float32)
        images /= 255
        
        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property    
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def ids(self):
        return self._ids
    
    @property
    def cls(self):
        return self._cls
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    @property 
    def num_examples(self):
        return self._num_examples
    
    def next_batch(self,batch_size):
        """
        returns a mini batch of examples from the given dataset
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            
            #epoch is finished
            self._epochs_completed += 1
            
            start = 0
            self._index_in_epoch = batch_size
            assert self._index_in_epoch <= self._num_examples
        end = self._index_in_epoch
        
        return self._images[start:end],self._labels[start:end],self._ids[start:end],self._cls[start:end]
    
    
      
   
        

def read_train_set(train_path,image_size,classes,validation_size = 0):
    
    class Datasets(object):
        pass
    data_sets = Datasets()
    
    images,labels,ids,cls = load_train(train_path,image_size,classes)
    print('it is loaded')
    images,labels,ids,cls = shuffle(images,labels,ids,cls)
    print('it is shuffled')
    if isinstance(validation_size,float):
        validation_size = int(validation_size * images.shape[0])
        
    valid_images = images[0:validation_size]
    valid_labels = labels[0:validation_size]
    valid_ids = ids[0:validation_size]  
    valid_cls = cls[0:validation_size] 

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]
    print('it is split into valid and train')
    data_sets.train = Dataset(train_images,train_labels,train_ids,train_cls)
    data_sets.valid = Dataset(valid_images,valid_labels,valid_ids,valid_cls)

    return data_sets

#train_path = "dataset/train/"        
#temp_data_sets = read_train_set(train_path,64,classes = ['cats','dogs'],validation_size = 0.3)
#temp_train = temp_data_sets.train
#temp_valid = temp_data_sets.valid
#
#temp_train_images = temp_train.images
#temp_valid_images = temp_valid.images
#temp_train_images.shape
#plt.figure(figsize = (10,50))
#for i in range(5):
#    plt.subplot(5,1,i+1)
#    plt.imshow(temp_train_images[i])
#    plt.title(temp_train.cls[i])
#    
#temp_valid_images.shape
#plt.imshow(temp_valid_images[0])
#plt.title(temp_valid.cls[0])


def read_test_set(test_path,image_size):
    X_test,X_test_id = load_test(test_path,image_size)
    return X_test,X_test_id

#
#test_path = "dataset/test"
#temp_images,temp_id = read_test_set(test_path,64)
#
#temp_images.shape