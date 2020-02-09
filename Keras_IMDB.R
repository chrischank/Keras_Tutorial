#This is a DL text classification tutorial using imdb dataset
library(keras)
library(tidyverse)

#This imdb dataset, the sequence of words have already been preprocessed and converted to sequence of integers
#num_words = 10000 keeps the top 10000 most frequently occuring word in the training data
imdb <- dataset_imdb(num_words = 10000)

#Create vector from variables of imdb
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels)%<-% imdb$test

#The dataset comes with an index mapping words to integers
#Which is downloaded separately

word_index <- dataset_imdb_word_index()

#################
#EXPLORE DA DATA#
#################

#The dataset comes preprocessed: each example is an array of integers representing the words of the movie review.
#Each label is an integer value of either 0/1
#Convert the texts of the reviews to integers, each integer represents a specifric word
paste0("Training entries: ", length(train_data), "labels: ", length(train_labels))
train_data[[1]]
length(train_data[[2]])

#Convert the integers back to words
#If we create a data frame from it, we can conveniently use it in both integers and words
word_index_df <- data.frame(
  word=names(word_index),
  idx=unlist(word_index, use.names=FALSE),
  stringsAsFactors = FALSE
)

#The first indices are reserved by +3, making it unique, so can be isolated
#mutate() = add new variables and preserves existing one
word_index_df <- word_index_df %>% mutate(idx=idx+3)

#Add new variables PAD, START, UNK, and UNUSED as 0,1,2,3
word_index_df <- word_index_df %>% 
  add_row(word="<PAD>", idx=0) %>%
  add_row(word="<START>", idx=1)%>%
  add_row(word="<UNK>", idx=2) %>%
  add_row(word="<USUSED>", idx=3)

#arrange() = order table rows by an expression involving its variables
word_index_df <- word_index_df %>% arrange(idx)

decode_review <- function(text){
  paste(map(text, function(number) word_index_df %>%
              filter(idx == number) %>%
              select(word)%>%
              pull()),
        collapse=" ")
}


#Now, we can decode_reivew() to display the text for the 420th review
decode_review(train_data[[420]])

#################
#PREPARE DA DATA#
#################

#The reviews (arrays of integers) must be converted to tensor before fed into the NN
#We can either One-hot-encode the arrays to convert them into c(0 AND 1),
  #The sequence[3, 5] would become a 10000 dim vector that is all 0 exceot for indices 3 and 5, which =1
  #Then make this 1st layer in our network (dense_layer)
  #This approach is memory intensive

#OR, we can pad the arrays so they all have the same length,
#then create an integer tensor of the shape num_example*max_length.
  #We can use an embedding layer capable of handling this shape as the 1st layer in our NN

#WE TAKE THE 2ND APPROACH
#pad_sequence() = transform a list of num_samples sequences (list of integers) into a matrix of shape (num_samples, num_timesteps)
train_data <- pad_sequences(
  train_data,
  value = word_index_df %>%  filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

test_data <- pad_sequences(
  test_data,
  value = word_index_df %>% filter(word == "<PAD>") %>% select(idx) %>% pull(),
  padding = "post",
  maxlen = 256
)

#Inspect the (now padded) first review:
train_data[1, ]


################
#BUILD DA MODEL#
################

#Input shape is the vocabulary count used for the movie reviews (10000 words)
vocab_size <- 10000

#The 1st layer is an embedding_layer, this layer takes integer encoded vocab and looks up the embedding vector for each word index.
  #These vectors are learned as the model trains
  #The vectors add a new dim to the output array, resulting in (batch, sequence, embedding)
#The 2nd layer is global_average_pooling_1d, layer returns a fixed-length output vector for each example by averaging over the sequence dimensions.
  #The fixed length output vector is piped through a dense_layer with 16 hidden units
#The last layer is densely connected with single output node.
  #Activation function = "sigmoid"

model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary

#COMPILE THE MODEL

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

#Create a validation set
  #When training, we want to check the accuracy of the model on data it hasn't seen before.
  #Create a validation set by setting apart 10000 examples from the original training data
x_val <- train_data[1:10000, ]
partial_x_train <- train_data[10001:nrow(train_data), ]

y_val <- train_labels[1:10000]
partial_y_train <- train_labels[10001:length(train_labels)]

#TRAIN DA MODEL 20 epochs, in mini-batches of n=512
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs=40,
  batch_size=512,
  validation_data=list(x_val, y_val),
  verbose=1
)

#EVALUATE THE MODEL
results <- model %>% evaluate(test_data, test_labels)
results

plot(history)

#The training loss decreases with each epoch, and teh training accuracy increases with each epoch
#That is expected when using gradient descent optimisation
#However, this isn't the case for validation loss and accuracy, which after 20 epochs peaked in accuracy and gain no more
  #Therefore this model over fits after 20ish epochs
  #To prevent overfitting, we could just stop training after 20 epochs or so
  #This is known as callback