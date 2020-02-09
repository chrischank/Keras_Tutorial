#This is a regression tutorial aiming to predict the output of a continuous value (e.g. price or probability)
#This exercise builds a model to predict the median price of homes in a Boston suburb during mid 1970s

library(keras)
library(tidyverse)

boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test

#This dataset has 506 total example
#We split this between 404 training and 102 testing:
paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

#This dataset contains 13 different features:
#Different input data are stored in different scale!
  #1. Per capita crime rate
  #2. Proportion of residential land zoned for lots over 25000 sqft
  #3. Proportion of non-retail business acres per town
  #4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
  #5. Nitric oxides concentration (parts per 10 million).
  #6. The average number of rooms per dwelling.
  #7. The proportion of owner-occupied units built before 1940.
  #8. Weighted distances to five Boston employment centers.
  #9. Index of accessibility to radial highways.
  #10. Full-value property-tax rate per $10,000.
  #11. Pupil-teacher ratio by town.
  #12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
  #13. Percentage lower status of the population.

train_data[1, ]

#Let's add column name for better data inspection
#Use tibble to convert vector into data.frame
library(tibble)

column_name <- c("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", 
                 "TAX", "PTRATIO", "B", "LSTAT")
train_df <- as_tibble(train_data)
colnames(train_df) <- column_name

train_df

train_labels[1:10]

#NORMALISE FEATURES
#It is recommended to normalise features that uses different scales and ranges
#Although the model might converge without feature normalisation
#It makes training difficult

#Test data is !not used when calculating the mean and st.dev

#Normalise training data
train_data <- scale(train_data)

#Use means and st.dev from training set to normalise the test set
#attr() = Get a specific attribute of an object
col_means_train <- attr(train_data, "scaled:center")
col_stdev_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center=col_means_train, scale=col_stdev_train)

#################
#CREATE DA MODEL#
#################

#Here we will use a sequential model with 2 densely connected hidden layer
#And an output layer that returns a single, continuous value
#THIS TIME, the model building steps are wrapped inside a function

build_model <- function(){
  
  model <- keras_model_sequential() %>%
    layer_dense(units=64, activation="relu",
                input_shape=dim(train_data)[2]) %>%
    layer_dense(units=64, activation="relu") %>%
    layer_dense(units=1)
  
  model %>% compile(
    loss="mse",
    optimizer=optimizer_rmsprop(),
    metrics=list("mean_absolute_error")
  )
  model
}

model <- build_model()
model %>% summary()

################
#TRAIN DA MODEL#
################

#This model gonna be trained for 500 epochs
#Here will alos show how to use a custom callback, replacing default training output by a single dot per epoch
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs){
    if (epoch %% 80 ==  0) cat("\n")
    cat(".")
  }
)

epochs <- 500

#Fit the model and store training stats

history <- model %>% fit(
  train_data,
  train_labels,
  epoch=epochs,
  validation_split=0.2,
  verbose=0,
  callbacks=list(print_dot_callback)
)

#Now we visualise the model's training progress using the metrics stored in the history variable
#We want to use this data to determine how long to train before the model stops making progress.

plot(history, metrics="mean_absolute_error", smooth=FALSE)+
  coord_cartesian(ylim = c(0, 5))

#The graph shows little improvement in the model after about 200 epochs.
#Let's update the fit method to auto stop training when the validation score does not improve.
#Using a callback function
#early_stop in callbacks() help halt the training
early_stop <- callback_early_stopping(monitor="val_loss", patience=20)

history <- model %>% fit(
  train_data,
  train_labels,
  epoch=epochs,
  validation_split=0.2,
  verbose=0,
  callbacks=list(early_stop, print_dot_callback)
)

plot(history, metrics="mean_absolute_error", smooth=FALSE)+
  coord_cartesian(xlim=c(0, 150), ylim = c(0, 5))

#Let's perform the trained model on test set
c(loss, mae) %<-% (model %>% evaluate(test_data, test_labels, verbose=0))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))

#########
#PREDICT#
#########

test_predictions <- model %>% predict(test_data)
test_predictions[ ,1]

#DIFFERENCE BETWEEN REGRESSION AND CLASSIFICATION NN
#MEAN SQURED ERROR (MSE) is a common loss function used for regression
#Simiilarly, evaluation metrics used for regression differ from classification.
  #A common regression metric is MEAN ABSOLUTE ERROR (MAE)
#If there is not much training data, prefer a small network with few hidden layers, and avoid OVERFITTING
#Early stopping and callback is a useful technique to prevent overfitting
