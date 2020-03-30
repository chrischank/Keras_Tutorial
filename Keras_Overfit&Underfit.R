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
first_text <- data.frame(word=1:10000, value=train_data[1, ])
(multi_hot_vec <- ggplot(first_text, aes(x=word, y=value))+
    geom_line()+
    theme(axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()))

#########################
#DEMONSTRATE OVERFITTING#
#########################

#Simplest way to demonstrate overfitting si to reduce the size of model
#i.e. the number of learnable parameters in the model (which is determined by the number of layers and the number of units per layer)
#In DL, the number of learnable parameters in a model is often referred to as the model's "capacity"
#A model with more parameters will have more "memorisation capacity"

#DL MODELS TEND TO BE GOOD AT FITTING TO THE TRAINING DATA,
#BUT THE REAL CHALLENGE IS GENERALISATION, NOT FITTING

#If the network has limited memorasation resources
#it will not be able to learn the mapping as easily
#To minimise tis loss, it will have to learn compressed representation that have more PREDICTIVE POWER
#If you make the model too small, it will have difficulty fitting to training data
#There's a balance between "too much capacity", and "not enough capacity"

#We will creat simple model using dense layer, then a smaller version, and compare them

#CREATE A BASELINE MODEL----

baseline_model <- keras_model_sequential() %>% 
  layer_dense(units=16, activation="relu", input_shape=10000) %>% 
  layer_dense(units=16, activation="relu") %>% 
  layer_dense(units=1, activation="sigmoid")

baseline_model %>%
  compile(optimizer="adam",
          loss="binary_crossentropy",
          metrics=list("accuracy"))

baseline_model %>% summary()

baseline_history <- baseline_model %>% 
  fit(train_data,
      train_labels,
      epochs=20,
      batch_size=512,
      validation_data=list(test_data, test_labels),
      verbose=2)

#CREATE A SMALLER MODEL----

smaller_model <- keras_model_sequential() %>% 
  layer_dense(units=4, activation="relu", input_shape=10000) %>% 
  layer_dense(units=4, activation="relu") %>% 
  layer_dense(units=1, activation="sigmoid")

smaller_model %>%
  compile(optimizer="adam",
          loss="binary_crossentropy",
          metrics=list("accuracy"))

smaller_model %>% summary()

#Train the model using the same data
smaller_history <- smaller_model %>% 
  fit(train_data,
      train_labels,
      epochs=20,
      batch_size=512,
      validation_data=list(test_data, test_labels),
      verbose=2)

#CREATE A BIGGER MODEL----
bigger_model <- keras_model_sequential() %>% 
  layer_dense(units=512, activation="relu", input_shape=10000) %>% 
  layer_dense(units=512, activation="relu") %>% 
  layer_dense(units=1, activation="sigmoid")

bigger_model %>% 
  compile(optimizer="adam",
          loss="binary_crossentropy",
          metrics=list("accuracy"))

bigger_model %>% summary()

#Train the model using the same data
bigger_history <- bigger_model %>% 
  fit(train_data,
      train_labels,
      epochs=20,
      batch_size=512,
      validation_data=list(test_data, test_labels),
      verbose=2)

#Plot the training and validation loss
#The smaller networks begins overfitting a little later than the baseline model
#It's performance degrades much slowly once it starts overfitting
#The larger network begins overfitting almost right away, significantly after 1 epoch
#The more capacity the network has, the quicker it will be able to model the training data (resulting in low training loss
#But the more susceptible it is to overfitting (resulting in large difference between the training and validation loss)

compare_cx <- data.frame(
  baseline_train=baseline_history$metrics$loss,
  baseline_val=baseline_history$metrics$val_loss,
  smaller_train=smaller_history$metrics$loss,
  smaller_val=smaller_history$metrics$val_loss,
  bigger_train=bigger_history$metrics$loss,
  bigger_val=bigger_history$metrics$val_loss
) %>% 
  rownames_to_column() %>% 
  mutate(rowname=as.integer(rowname)) %>% 
  gather(key="type", value="value", -rowname)

(losses <- ggplot(compare_cx, aes(x=rowname, y=value, color=type))+
    geom_line()+
    xlab("epoch")+
    ylab("loss"))

############
#STRATEGIES#
############

#ADD WEIGHT REGULARISATION----

#OCCAM'S RAZOR: Given 2 explanations for something, the explanation most likely to be correct is the "simplest" one
  #The one that makes the least amount of assumptions
#Given some training data, and a network architecture
#There are multiple set of weights values that could explain the data
#Simpler models are less likely to overfit than complex ones

#A "simple model" in this context is a model where distribution of parameters values has less ENTROPY
#Common way of overfit mitigation is to put constraint on the complexity of a network
#By forcing its weights to only take on small values, which makes the distribution of weight values more "regular"

#COMES IN 2 LAYERS:

#L1 regularisation, where the cost is added proportional to the absolute value of weights coefficients
#i.e. to what is called the "L1 norm" of the weights

#L2 regularisation, where the cost added is proportional to the square of the value of the weights coefficient
#i.e. to what is called the "L2 norm" of the weights
#L2 regularisation is also called weight decay in neural networks

l2_model <- keras_model_sequential() %>% 
  layer_dense(units=16, activation="relu", input_shape=10000,
              kernel_regularizer=regularizer_l2(l=0.001)) %>% 
  layer_dense(units=16, activation="relu",
              kernel_regularizer=regularizer_l2(l=0.001)) %>% 
  layer_dense(units=1, activation="sigmoid")

l2_model %>%
  compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=list("accuracy"))

l2_history <- l2_model %>% 
  fit(train_data,
      train_labels,
      epochs=20,
      batch_size=512,
      validation_data=list(test_data, test_labels),
      verbose=2)

#l2(0.001) means that every coefficient in the weight matrix of layer will ad 0.001
compare_cx <- data.frame(
  baseline_train=baseline_history$metrics$loss,
  baseline_val=baseline_history$metrics$val_loss,
  l2_train=l2_history$metrics$loss,
  l2_val=l2_history$metrics$val_loss
) %>% 
  rownames_to_column() %>% 
  mutate(rowname=as.integer(rowname)) %>% 
  gather(key="type", value="value", -rowname)

(l2base_losses <- ggplot(compare_cx, aes(x=rowname, y=value, color=type))+
    geom_line()+
    xlab("epoch")+
    ylab("loss"))

#As we can see the L2 regularised model become much more resistant to overfitting
#eventhough both models have the same number of parameters

#############
#ADD DROPOUT#
#############

#Dropout is another commonly used technique for NN
#Let's add 2 dropout layers in our IMDB network to see how well they do at reducing overfitting

dropout_model <- keras_model_sequential() %>% 
  layer_dense(units=16, activation="relu", input_shape=10000) %>% 
  layer_dropout(0.6) %>%
  layer_dense(units=16, activation="relu") %>% 
  layer_dropout(0.6) %>% 
  layer_dense(units=1, activation="sigmoid")

dropout_model %>% 
  compile(optimizer="adam",
          loss="binary_crossentropy",
          metrics=list("accuracy"))

dropout_history <- dropout_model %>% 
  fit(train_data,
      train_labels,
      epochs=20,
      batch_size=512,
      validation_data=list(test_data, test_labels),
      verbose=2)

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  dropout_train = dropout_history$metrics$loss,
  dropout_val = dropout_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

(dropout_loss <- ggplot(compare_cx, aes(x=rowname, y=value, color=type))+
    geom_line()+
    xlab("epoch")+
    ylab("loss"))

#Adding a dropout is a clear improvement
