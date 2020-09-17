# Tensorflow_Learning_Yuxia
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
1. Cats v Dogs full Kaggle Challenge exercise. 
2. Add Augmentation to it, and experiment with different parameters to avoid overfitting. 
3. Transfer Learning to increase training accuracy for Horses v Humans. To avoid crazy overfitting, your validation set accuracy should be around 95%. 
4. Going from binary classification to multi class classification. In this one you'll use the Sign Language dataset from https://www.kaggle.com/datamunge/sign-language-mnist.

### Learning
* Prepare the training and testing image in each directory
* Set up the training and validation generator to perform preprocessing: rescale, data augumentation.
* flow_from_directory and from slice
* Transfer learning
* Multi-class classifer
* Plot_model to visualize the model: Inception network visualization

/*To do:*/
loss='binary_crossentropy': Check different loss definition
Batch normalization layer parameter 
Customized the layer object


## Course 3 Natural Language Processing in TensorFlow
1. Turning the sentence into sequences via TFIDF, bag of words, word counts. Learn tokenizer(word index) and pad_sequences APIs in TensorFlow to build a sarcasm classifer. 
2. Use word embedding to build a neural netowrks with the usage of semantics of words. The embedding is trained on the fly with the model training. Words with similar embedding and semantics tends to cluster together.
3. Use the embedding from transfer learning: GloVe. Fine tune the network with a two stage process: classifer header and unforze a few top layers. 
4. Sentiment can also be determined by the sequence in which words appear. LSTM, GRU, Conv1D combined with embedding to model the sequence. 
5. Language modeling: word based and character based text generator. Data preparation for both cases. The benefit for character based is that the number of output class is small(e.g.: 63) instead of the size of the vocabulary. 

### Learning
* LSTM(unit=64, return_sequence=True): 64 is the dimesion of the final output. It do not need to know the internal number of steps. 

## Course 4 Sequences, Time Series and Prediction
1. NN in time series: anomaly detection, imputation and forecasting. Statistical forecasting: Moving average do not model trend. Therefore, use differencing to remove the trend before applying MA. 
