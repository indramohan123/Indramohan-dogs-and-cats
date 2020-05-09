# Dogs_vs_cats_cnn_tensorflow
Dogs_vs_cats dataset is used which is publicly available on kaggle
The project contains three files.
Preprocessor.py is used for preprocessing the data, and loading the data images from folders, and applying processes like shuffling, converting them to arrays, and splitting the training data into validation data.
LayersConstructor.py is used to create tensorflow layers, which makes the use of tf.nn library
CNN.py makes the use of the above mentioned layers to construct a model which will clasify the images as a cat or a dog.
