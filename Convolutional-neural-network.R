## -----------------------------------------------------------
options(warn = -1)


## -----------------------------------------------------------
knitr::purl("Convolutional-neural-network.Rmd")


## -----------------------------------------------------------
pacman::p_load(tidyverse, keras, tensorflow)


## -----------------------------------------------------------
cifar100 <- keras::dataset_cifar100()


## -----------------------------------------------------------
attributes(cifar100)


## -----------------------------------------------------------
train_x <- cifar100$train$x
train_z <- cifar100$train$y

test_x <- cifar100$test$x
test_z <- cifar100$test$y


## -----------------------------------------------------------
dim(train_x)
dim(test_x)

scale <- range(train_x[1,,,1])[2]

# output images 100 unique
range(train_z)


## -----------------------------------------------------------
train_x <- train_x / scale
test_x <- test_x / scale


## -----------------------------------------------------------
train_y <- to_categorical(train_z)
test_y <- to_categorical(test_z)


## -----------------------------------------------------------



## -----------------------------------------------------------
library(jpeg)

set.seed(1)
index <- sample(seq(5000), 10)
for(i in index){
  plot(as.raster(train_x[i,,,]),asp = .5)
  
}



## -----------------------------------------------------------
model <- keras_model_sequential() |> 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                padding = "same", activation = "relu",
                input_shape = c(32,32,3)) |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_flatten() |> 
  layer_dropout(rate=.6) |> 
  layer_dense(units = 512, activation = "relu") |> 
  layer_dense(units = 100, activation = "softmax")
  



## -----------------------------------------------------------
model |> 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_rmsprop(),
          metrics = c("accuracy"))


## -----------------------------------------------------------
tictoc::tic()
train_fit <- model |>  
  fit(train_x, train_y, epoch = 150, batch_size = 256, 
      validation_split = .2)
stop <- tictoc::toc()


## -----------------------------------------------------------
model |>
  evaluate(test_x, test_y)


## -----------------------------------------------------------
pred_prob <- model |> 
 predict(test_x)

pred <- (max.col(pred_prob) -1 ) |> 
  as.matrix(ncol = 1)


## -----------------------------------------------------------
library(caret)


vec_true <- as.vector(test_z) |> 
  factor(levels = 0:99)
#vec_true

vec_pred <- as.vector(pred) |> 
  factor(levels = 0:99)
#vec_pred



confusionMatrix(data = vec_pred, reference = vec_true)



## -----------------------------------------------------------
set.seed(1)
index <- sample(1:dim(cifar100$train$x)[1],replace = FALSE,size = 40000)

# normalize data
scale <- range(cifar100$train$x[1,,,1])[2]

train_x_aug <- cifar100$train$x[index,,, ] / scale
train_z_aug <- cifar100$train$y[index, ] |> matrix(ncol = 1)

val_x_aug <- cifar100$train$x[-index,,, ] / scale
val_z_aug <- cifar100$train$y[-index, ] |> matrix(ncol = 1)

test_x_aug <- cifar100$test$x /scale
test_z_aug <- cifar100$test$y



## -----------------------------------------------------------
train_y_aug <- to_categorical(train_z_aug)
val_y_aug <- to_categorical(val_z_aug)
test_y_aug <- to_categorical(test_z)


## -----------------------------------------------------------

aug_train <- image_data_generator(
  rotation_range = 4, 
  zoom_range = 0.3,
  shear_range = 0.1,
  width_shift_range = 0.3,
  height_shift_range = 0.25,
  horizontal_flip = T,
  fill_mode = "nearest"
)

# Apply data augmentation to the images
augmented_data <- flow_images_from_data(
  train_x_aug,
  train_y_aug,
  generator = aug_train,
  batch_size = 256,
  shuffle = TRUE          
)


val_generator <- flow_images_from_data(
  val_x_aug,
  val_y_aug,
  batch_size = 256,
  shuffle = FALSE
)


## -----------------------------------------------------------
# model
model1 <- keras_model_sequential() |> 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), 
                padding = "same", activation = "relu",
                input_shape = c(32,32,3)) |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), 
                padding = "same", activation = "relu") |> 
  layer_max_pooling_2d(pool_size = c(2,2)) |> 
  layer_flatten() |> 
  layer_dropout(rate=.6) |> 
  layer_dense(units = 512, activation = "relu") |> 
  layer_dense(units = 100, activation = "softmax")


## -----------------------------------------------------------
model1 |> 
  compile(loss = "categorical_crossentropy",
          optimizer = optimizer_rmsprop(), metrics = c("accuracy"))




## -----------------------------------------------------------
model1 %>% fit(
  augmented_data,
  epochs = 150,
  batch_size = 256,
  validation_data = val_generator
  
)




## -----------------------------------------------------------
model1 |>
  evaluate(test_x_aug, test_y_aug)


## -----------------------------------------------------------
range(test_z)


## -----------------------------------------------------------
pred_prob <- model1 |> 
 predict(test_x_aug)

pred <- (max.col(pred_prob) -1 ) |> 
  as.matrix(ncol = 1)


## -----------------------------------------------------------
library(caret)

vec_true <- as.vector(test_z_aug) |> 
  factor(levels = 0:99)

vec_pred <- as.vector(pred) |> 
  factor(levels = 0:99)

confusionMatrix(data = vec_pred, reference = vec_true)


