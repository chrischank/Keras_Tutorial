#Keras is an API built upon TensorFlow, it also run CNTK, or Teano
install.packages("keras")
library(keras)
library(tidyverse)
install_keras()

#Recognising handwritten digits from MNIST dataset
#Consist 

#Preparing the Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

#x data is 3D array (image, width, height)
#We need to convert the 3D arrays into matrices by reshaping the width and height into a single dimension
#Flattening a 28x28 into length 784 vector)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
#rescale
x_train <- x_train/255
xtest <- x_test/255

#y data is an integer vecto with values 0:9, to prepare, change the vectors into binary class matrices
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

#DEFINING THE MODEL
#Sequential model, a linear stack of layers
#Softmax activation function takes as input a vector K real numbers, and normalises it into probability distribution consisting of K probabilities
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation="relu", input_shape=c(784)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units=10, activation="softmax")

summary(model)

#Compile the model with appropriate loss function, optimiser, and metrics:
model %>% compile(
  loss="categorical_crossentropy",
  optimizer=optimizer_rmsprop(),
  metrics=c("accuracy")
)

#TRAINING & EVALUATION
#Use fit() to train the model for 30 epochs suing batches of 128 images
history <- model %>% fit(
  x_train, y_train,
  epochs=30, batch_size=128,
  validation_split=0.2
)

plot(history)
