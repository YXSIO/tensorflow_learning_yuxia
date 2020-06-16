# tensorflow_learning_yuxia
This repo contains the notebooks for deep learning AI crash course on Tensorflow.

## Course 1 Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
1. House price with linear model - single neuron. 
2. Hand writing recognition with fully connected layers.
3. Improve MNIST with convolutions.
4. Image classifier for complex images.Create a classifier for a set of happy or sad images with callback to cancel training once accuracy is greater than 0.99.

### Learning
* input_shape definition
* layer object: method and attribute
* Kernel and bias: kernel can be thought as weights
* Implement helper function to calculate the number of parameter for various layers
* Implement training as function


## Course 2 Convolutional Neural Networks in TensorFlow
1. it’s time to go back to the Cats v Dogs full Kaggle Challenge exercise. Last week you built a classifier for Cats v Dogs and trained it for a few epochs. 
2. Now that you’ve seen Image Augmentation in action, This week I want you to add Augmentation to it, and experiment with different parameters to avoid overfitting. 
3. This week your exercise will be to apply what you've learned about Transfer Learning to see if you can increase training accuracy for Horses v Humans. To avoid crazy overfitting, your validation set accuracy should be around 95% if you do it right! Your training should automatically stop once it reaches this desired accuracy.
4. Now that you've explored the concepts behind going from binary classification to multi class classification, it's time for another Exercise. In this one you'll use the Sign Language dataset from https://www.kaggle.com/datamunge/sign-language-mnist.

### Learning
* Prepare the training and testing image in each directory
* Set up the training and validation generator to perform preprocessing: rescale, data augumentation.
* flow_from_directory and from slice
* Transfer learning

## Course 3 Natural Language Processing in TensorFlow


## Course 4 Sequences, Time Series and Prediction
