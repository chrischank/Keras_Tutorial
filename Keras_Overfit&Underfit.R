##############################
#OVERFITTING AND UNDERFTITING#
##############################

#We saw previously, accuracy of model on validation data
#Would peak after training for a number of epochs, then start decrease
#THIS IS OVERFITTING
#What we want to do is develop models that generalse well to testing data
#The opposite of overfitting is underfitting
#UNDERFITTING = Occurs when there is still room for improvement on the test data
  #Reasons: If the model is not powerful enough, is over-regularised, or has simply not been trained long enough
#UNDERFITTING means the network has not learned the relevant patterns

#If you train for too long, it'll OVERFIT
#OVERFITTING learn patterns from the training data that don't generalise to test data
#To prevent overfitting, use more training data
#If thats not possible, use technique like regularisation
#Regularisation places constraints on quantity and type of info your model can store

library(keras)
library(tidyverse)

num_words <- 10000
imdb <- dataset_imdb(num_words = num_words)

c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test

#Rather than using an embedding , here we will multi-hot encode the sentences.
#This model will quickly overfit to the training set.
#Multi-hot-encoding = Turning them into vectors of 0s and 1s
#This would mean for instance turning the sequence [3, 5] into a 10000 dim vector
#That will be all 0 except for indices 3 and 5

multi_hot_sequences <- function(sequences, dimension){
  multi_hot <- matrix(0, nrow=length(sequences), ncol=dimension)
  for (i in 1:length(sequences)){
    multi_hot[i, sequences[[i]]] <- 1
  }
  multi_hot
}

train_data <- multi_hot_sequences(train_data, num_words)
test_data <- multi_hot_sequences(test_data, num_words)

#Let's look at 1 of the resulting multi-hot vectors

