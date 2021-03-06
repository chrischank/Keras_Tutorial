library(keras)
library(tidyverse)
install_keras()

#In this tutorial, we train a NN to classify clothing (e.g. sneakers and shirts)
#Using 60000 images to train the network and 10000 images to evaluate the accuracy
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

#We now have 4 arrays: train_images & train_labels are training set | test_images & test_labels are evaluation
#Our images are 28x28 array, pixel range 0:255(16 bits). The labels are arrays of integers, ranging from 0:9
#To map each image to a single label, we first need to create the class name, store it as vector for later use
class_names=c("T-shirt",
              "Trousers",
              "Pullover",
              "Dress",
              "Coat",
              "Sandal",
              "Shirt",
              "Sneaker",
              "Bag",
              "Ankle boot")

#Explore the data with dim command, which retrieve the dimension of an object
dim(train_images)
train_labels[1:20]
dim(test_images)
dim(test_labels)

####################
#PREPROCESS DA DATA#
####################

#Da data must be preprocessed before training the network.
image_1 <- data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x=x, y=y, fill=value))+
  geom_tile()+
  scale_fill_gradient(low="white", high="black", na.value=NA)+
  scale_y_reverse()+
  theme_minimal()+
  theme(panel.grid=element_blank())+
  theme(aspect.ratio=1)+
  xlab("")+
  ylab("")

#We need to scale these value to a range of 0 and 1 before feeding to the NN model, we simply divide by 255 (16 bits)
train_images <- train_images/255
test_images <- test_images/255


#Display the 1st 25 images from the training set, display the class name below the image
#Verify that the data is in the correct format --> ready to build and train the network
par(mfcol=c(5, 5))
par(mar=c(0, 0, 1.5, 0), xaxs="i", yaxs="i")
for(i in 1:25){
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col=gray((0:255)/255), xaxt="n", yaxt="n",
        main=paste(class_names[train_labels[i]+1]))
}

################
#BUILD DA MODEL#
################

#Setup the layers, layers extract representation from the data
#Most of deep learning consists of chaining together simple layers, most layers, like layer_dense have parameters that are learned during training
#layer_flatten transforms the format of the images from 2D array (28x28),
#to a 1D array of 28*28=784
#layer flatten have no parameters to learn, it's just a reformat layer
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units=128, activation="relu") %>%
  layer_dense(units=10, activation="softmax")

#Compile the model
#Loss function <- measures how accurate the model is during training, want to minimise this
#Optimiser <- this is how the model is updated based on the data it sees and its loss function
#Metrics <- Used to monitor the training and testing steps. This example uses accuracy, the fraction of images that are correctly classified
model %>% compile(
  optimizer="adam",
  loss="sparse_categorical_crossentropy",
  metrics=c("accuracy")
)

################
#TRAIN DA MODEL#
################

#The model will now learn to associate images and labels
#We ask the model to make predictions about a test set
model %>% fit(train_images, train_labels, epochs=5)

#Evaluate accuracy
score <- model %>% evaluate(test_images, test_labels)

cat("Test loss:", score$loss, "\n")
cat("Test accuracy", score$acc, "\n")
#TEST ACCURACY RESULT IS 0,8523, which means that the accuracy on the test dataset is a little less than the accuracy on the training dataset.
#THIS IS AN EXAMPLE OF OVERFITTING <- OVERFITTING is when ML model performs worse on new data than on the training data

#################
#MAKE PREDICTION#
#################

#Now that the model is trained, we can use it to make prediction
#A prediction is an array of 10 numbers, these describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing
predictions <- model%>%predict(test_images)
predictions[1, ]
#which.max tells me which class have the highest accuracy result, in this case its [10], which means the algo best predict "Ankle boot"
which.max(predictions[1, ])

#Alternatively, we can also directly get class prediction:
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]

test_labels[1]

#Let's plot several images with their prediction. Correct prediction labels are green and incorrect prediction labels are red
par(mfcol=c(5, 5))
par(mar=c(0, 0, 1.5, 0), xaxs="i", yaxs="i")
for(i in 1:25){
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  #subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ])-1
  true_label <- test_labels[i]
  if(predicted_label == true_label){
    color <- "green"
  } else {
    color <- "red"
  }
  image(1:28, 1:28, img, col=gray((0:255)/255), xaxt="n", yaxt="n",
        main=paste0(class_names[predicted_label+1],"(",
                    class_names[true_label+1], ")"),
        col.main=color)
}

#Finally, we can use the trained model to make a prediction about a single image
img <- test_images[1, , , drop=FALSE]
dim(img)

predictions <- model %>% predict(img)
predictions

#subtract 1 as labels are 0-based
prediction <- predictions[1, ]-1
which.max(prediction)

class_pred <- model %>% predict_classes(img)
class_pred
