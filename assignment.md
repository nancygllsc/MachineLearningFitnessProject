# Practical Machine Learning -  Prediction Assignment
  
# Introduction

This is an analysis for final assignment of the Coursea course 'Practical Machine Learning'.

http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har



#Executive summary

The gradient boosted machine  (0.9998) achieved a better accuracy than the random forsest (0.9997)


# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. 
These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. 



## Goal

Predict the manner in which a participant did the exercise (class A - properly - or any other incorrect way)

# Data
## Obtaining Data
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
save_file = function(url, name) {
    if (!file.exists(name)) {
        library(downloader)
        download(url, destfile = name)
    }
}

save_file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
save_file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")


string40 <- "ncnnccnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"
string80 <- "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"
string120 <- "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"
string160 <- "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnc"
colString <- paste(string40, string80, string120, string160, sep = "")

data.training <- readr::read_csv("pml-training.csv", col_names = TRUE, col_types = colString)
data.testing <- readr::read_csv("pml-testing.csv", col_names = TRUE, col_types = colString)
data.training <- as.data.frame(data.training)
```
## Data Exploration

The goal of this project is to predict the manner in which they did the exercise. 
This is the "classe" variable in the training set, the last column. Let's have a look at the training data. 
Both datasets have 160 rows with the rraining set having 160 observations and the testing having 20 observations.


```r
dim(data.training)
```

```
## [1] 19622   160
```

```r
dim(data.testing)
```

```
## [1]  20 160
```

The 'classe' variable is the indicator of the training outcome.
Classe 'A' corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. 
Below shows a plot of the distribution of this variable throughout the training set.
![](assignment_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

## Cleaning Data
To clean up the data we remove columns where ALL values are NA


```r
data.training <- data.training[, colSums(is.na(data.training)) == 0]
data.testing <- data.testing[, colSums(is.na(data.testing)) == 0]
```

## Training 
#### split the original training set because original test set does not have enough observations

```r
data.include <- createDataPartition(data.training$classe, p = .70, list = FALSE)
data.train <- data.training[data.include,]
data.test <- data.training[-data.include,]
```

# Model build - Random Forest

For this random forest model, we apply cross validation: the data is being splitted into five parts, each of them taking the role of a validation set once. A model is built five times on the remaining data, and the classification error is computed on the validation set. The average of these five error rates is our final error. This can all be implemented using the caret train function. We set the seed as the sampling happens randomly.

```r
cat("Random Forest model started")
```

```
## Random Forest model started
```

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl.rf <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)
timer.start <- Sys.time()
model.rf <- train(classe ~ ., data = data.train, method = "rf", trControl = fitControl.rf, verbose = FALSE, na.action = na.omit)
timer.end <- Sys.time()
stopCluster(cluster)
registerDoSEQ()
paste("Random Forest took: ", timer.end - timer.start, attr(timer.end - timer.start, "units"))
```

```
## [1] "Random Forest took:  2.51922355095545 mins"
```

# Prediction - Random Forest

```r
prediction.rf <- predict(model.rf, data.test)
confusion_matrix.rf <- confusionMatrix(prediction.rf, data.test$classe)
confusion_matrix.rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    1 1138    0    0    0
##          C    0    1 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9996     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9991   1.0000   1.0000   1.0000
## Specificity            1.0000   0.9998   0.9998   1.0000   1.0000
## Pos Pred Value         1.0000   0.9991   0.9990   1.0000   1.0000
## Neg Pred Value         0.9998   0.9998   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1934   0.1743   0.1638   0.1839
## Detection Prevalence   0.2843   0.1935   0.1745   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9995   0.9999   1.0000   1.0000
```

# Model build - Gradient Boosting Machine
Now we will do exactly the same, but use boosting instead of random forests. Getting the accuracy, predications... works with the same code.

```r
cat("Gradient Boosting Machine model started")
```

```
## Gradient Boosting Machine model started
```

```r
fitControl.gbm <- trainControl(method="cv",number=5,allowParallel=TRUE)
timer.start <- Sys.time()
model.gbm <- train(classe ~ ., data = data.train, method = "gbm", trControl = fitControl.gbm, verbose = FALSE, na.action = na.omit)
timer.end <- Sys.time()
paste("Gradient Boosting Machine took: ", timer.end - timer.start, attr(timer.end - timer.start, "units"))
```

```
## [1] "Gradient Boosting Machine took:  3.89540884892146 mins"
```
  
# Prediction - Gradient Boosting Machine

```r
cat("Gradient Boosting Machine predictions")
```

```
## Gradient Boosting Machine predictions
```

```r
prediction.gbm <- predict(model.gbm, data.test)
confusion_matrix.gbm <- confusionMatrix(prediction.gbm, data.test$classe)
confusion_matrix.gbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    1 1138    0    0    0
##          C    0    1 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9996     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9991   1.0000   1.0000   1.0000
## Specificity            1.0000   0.9998   0.9998   1.0000   1.0000
## Pos Pred Value         1.0000   0.9991   0.9990   1.0000   1.0000
## Neg Pred Value         0.9998   0.9998   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1934   0.1743   0.1638   0.1839
## Detection Prevalence   0.2843   0.1935   0.1745   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9995   0.9999   1.0000   1.0000
```


