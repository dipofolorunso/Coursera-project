The goal of the project
-----------------------

This goal of this project is to use accelerometer training data to predict the manner in which six participants did the exercise using the test data.


The training of the model
-------------------------

First read the data:

```r
rawData <- read.csv("pml-training.csv", na.strings = c("NA", ""))
```

### Reduce the dataset

First check the proportion of missing values.


```r
propNAs <- colMeans(is.na(rawData))
table(propNAs)
```

```
## propNAs
##                 0 0.979308938946081 
##                60               100
```

In 100 columns 97.93% are missing. These columns will be removed. Only the columns without any `NA`s will be kept.


```r
colNA <- !propNAs
rawDataReduced <- rawData[colNA]
```

The column `X` contains the row numbers. The column `user_name` contains the name of the user. These variables cannot be usefull as predictors and therefore they are removed. Also, the three columns containing time stamps (`raw_timestamp_part_1`, `raw_timestamp_part_2`, and `cvtd_timestamp`) will not be used. The factors `new_window` and `num_window` will be removed because they are not relevent to the prediction.


```r
notSensor <- grep("^X$|user_name|timestamp|window", names(rawDataReduced))
rawDataReduced2 <- rawDataReduced[-notSensor]
```


### Preparing the data for training

Using the function `createDataPartition` of the `caret` package, the data is split into a training and a cross-validation data set in the ratio 70:30. 


```r
library(caret)
```

```r
inTrain <- createDataPartition(y = rawDataReduced2$classe, p = 0.7, list = FALSE)
```



```r
training <- rawDataReduced2[inTrain, ]
nrow(training)
```

```
## [1] 13737
```

```r
crossval <- rawDataReduced2[-inTrain, ]
nrow(crossval)
```

```
## [1] 5885
```


### Train a model

Using the *random-forest* technique to generate a predictive model, 


```r
library(randomForest)
```


```r
trControl <- trainControl(method = "cv", number = 2)
modFit <- train(classe ~ ., data = training, method = "rf", prox = TRUE, trControl = trControl)
```

### Evaluate the model (out-of-sample error)

The final model was used to predict the outcome in the cross-validation dataset.


```r
pred <- predict(modFit, newdata = crossval)
```

The function `confusionMatrix` was used to calculate the accuracy of the prediction.


```r
confMat <- confusionMatrix(pred, reference = crossval$classe)
accuracy <- confMat$overall["Accuracy"]
accuracy
```

```
##  Accuracy 
## 0.9923534
```

The accuracy of the prediction is 99.24%. Hence, the *out-of-sample error* is 0.76%.


### Variable importance

The five most important variables in the model and their relative importance values are:


```r
Imp <- varImp(modFit)$importance
Imp[head(order(unlist(vi), decreasing = TRUE), 5L), , drop = FALSE]
```

```
##                     Overall
## roll_belt         100.00000
## pitch_forearm      60.51577
## yaw_belt           51.70924
## magnet_dumbbell_y  46.05269
## magnet_dumbbell_z  44.17784
```

***************************************************************************

