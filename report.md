#  Qualitative Human Activity prediction in Fitness

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, data used is from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:

+ Class A: exactly according to the specification
+ Class B: throwing the elbows to the front
+ Class C: lifting the dumbbell only halfway
+ Class D: lowering the dumbbell only halfway
+ Class E: throwing the hips to the front

Refer to http://groupware.les.inf.puc-rio.br/har for more details.

## Goal
This project is a part of the Coursera Practical Machine Learning course. The goal is to predict the manner in which people did the exercise. This is the "classe" variable in the training set.

## Report
This report includes:
+ Cleaning the data for preprossing; 
+ Preprocessing the data;
+ Different models for prediction;
+ Results of prediction model predicting 20 different test cases. 

### Libraries used
Below are the various R packages that were used.

    # libraries
    library(caret)
    library(randomForest)
    library(rpart)
    library(rpart.plot)
    library(ggplot2)
    library(lattice)
    library(rattle)

### Importing the data
Load the training and test data sets from the URLs. 
na.strings: a character vector of strings which are to be interpreted as NA values

    trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    traindata <- read.csv(url(trainUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))
    testdata <- read.csv(url(testUrl), header=TRUE, sep=",", na.strings=c("NA","#DIV/0!",""))

### Preparing data for preprocessing
We need to inspect and clean the data for further processing.

**Inspecting the different features of the data**

dim(traindata)
summary(traindata)
dim(testdata)
summary(testdata)
    
It can be noticed that some columns don't contain usefull data and there are many columns with missing data. Some variables, like stddev_, avg_ seem to be derived from the original data.
Although the number of columns of both training and test data is equal, the last column of the test set is different (problem_id) from the last column of the training data (classe). We need to make sure to apply the same transformations on both the training and test data.


**Removing columns with not much useful information** 

    # eliminating first 7 columns like userid and timestamps as they are not useful to predict the class of the activity  
    traindata <- traindata[,-seq(1:7)]
    testdata <- testdata[,-seq(1:7)]

**Removing columns with NAs**
This reduces the number of features

    # selecting the columns that don't have NAs
    iNA <- as.vector(sapply(traindata[,1:152],function(x) {length(which(is.na(x)))!=0}))
    traindata <- traindata[,!iNA]
    testdata <- testdata[,!iNA]

**Reducing the number of variables by removing the highly correlated variables**

There can be highly correlated variables which could reduce the performance of a model. Hence they'll need to be excluded.

    # last column is the column of interest. Get the index of the last column.
    len <- as.numeric(ncol(traindata))
    
    # setting the variables to numerics to check for correlation (except the last variable)
    for (i in 1:len-1) {
    traindata[,i] <- as.numeric(traindata[,i])
    testdata[,i] <- as.numeric(testdata[,i])
    }
    
    # Checking the correlations and doing levelplot 
    check <- cor(traindata[, -c(len)])
    diag(check) <- 0 
    plot( levelplot(check, 
                    main ="Correlation matrix",
                    scales=list(x=list(rot=90), cex=1.0),))
                    
![Correlation Matrix](/Corrmat.png)

    # Extracting only the highly correlated variables with a cutoff of 90%
    highcorr <- findCorrelation(cor(traindata[, -c(len)]), cutoff=0.9)
    
    # Removing the highly correlated variables from both training and test data
    traindata <- traindata[, -highcorr]
    testdata <- testdata[, -highcorr]

**Preproccesing of the variables**

The number of variables has been reduced to 46. We still need to preprocess the data.
The knnImpute method uses the most similar (nearest) neighbors to impute the missing values using the average value of its neighbors for that column. Hence knnImpute method is used to  impute the missing values in the data along with centering and scaling.
Centering is done by subtracting the column means (omitting NAs) of x from their corresponding columns.
Scaling is done by dividing the (centered) columns of x by their standard deviations.

    # Preprocessing and getting the prediction.
    pretrain <-preProcess(traindata[,1:len-1],method=c('knnImpute', 'center', 'scale'))
    trainPred <- predict(pretrain, traindata[,1:len-1])
    trainPred$classe <- traindata$classe
    
    pretest <-predict(pretrain,testdata[,1:len-1])
    pretest$problem_id <- testdata$problem_id

**Checking and removing the variables whose variance are close to zero**
Variables with nearly zero variance have less prediction value, so we need to remove them. Although in this case, there are none to be elimnated.

    # checking for near zero variance
    trainNZV <- nearZeroVar(trainPred, saveMetrics=TRUE)
    if (any(trainNZV$nzv)) nzv else message("No such variables")
    trainPred <- trainPred[,trainNZV$nzv==FALSE]
    pretest <- pretest[,trainNZV$nzv==FALSE]

### Cross validation
To train and test the model, we need to create sets for training and a testing from the complete training data.

    # split dataset into training and test set
    trainidx <- createDataPartition(y=trainPred$classe, p=0.7, list=FALSE )
    trainset <- trainPred[trainidx,]
    testset <- trainPred[-trainidx,]


    
### Model 1: Decision Tree
Starting with a simple Decision Tree for classification. 

    # Set a seed for reproducability.
    set.seed(12345)

    # Decision Tree using rpart
    decisionTree <- rpart(classe ~ ., data=trainset, method="class")
    
    # Visualizing the decisionTree using fancyRpartPlot
    fancyRpartPlot(decisionTree)

![Decision Tree Plot](/decisionTree.png)
    
### Accuracy of Model 1 on training set and cross validation set
Accuracy obtained is 0.7206

    # cross validation
    pred1 <- predict(dt, testset, type = "class")
    confusionMatrix(pred1, testset$classe)
    
    # Confusion Matrix and Statistics
             # Reference
    # Prediction    A    B    C    D    E
            # A 1513  227   30  106   77
            # B   61  623  146   49  124
            # C   39  142  743   72  135
            # D   51  112   91  681   65
            # E   10   35   16   56  681

  # Overall Statistics
                                             
                  # Accuracy : 0.7206         
                    # 95% CI : (0.709, 0.7321)
      # No Information Rate : 0.2845         
       # P-Value [Acc > NIR] : < 2.2e-16      
                                             
                     # Kappa : 0.6447         
    # Mcnemar's Test P-Value : < 2.2e-16      

Statistics by Class:

                        # Class: A Class: B Class: C Class: D Class: E
   # Sensitivity            0.9038   0.5470   0.7242   0.7064   0.6294
   # Specificity            0.8955   0.9199   0.9201   0.9352   0.9756
   # Pos Pred Value         0.7747   0.6211   0.6569   0.6810   0.8534
   # Neg Pred Value         0.9591   0.8943   0.9405   0.9421   0.9212
   # Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
   # Detection Rate         0.2571   0.1059   0.1263   0.1157   0.1157
   # Detection Prevalence   0.3319   0.1704   0.1922   0.1699   0.1356
   # Balanced Accuracy      0.8997   0.7335   0.8222   0.8208   0.8025
    



### Model 2: Random Forest
Tree Ensembles could provide better accuracy than simple decision tree. Hence using the tuneRF function to calculate the optimal mtry and use that in a random forest function.
   
    
    # Getting the best mtry
    bestmtry <- tuneRF(trainset[-len],trainset$classe, ntreeTry=100, 
                       stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
    
    mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]
  
    # Random Forest model
    rf <-randomForest(classe~.,data=trainset, mtry=mtry, ntree=501, 
                          keep.forest=TRUE, proximity=TRUE, 
                          importance=TRUE,test=trainset)
    

### Results of Model 2
Examining Out-Of-Bag (OOB) error-rate. Looking at the mean decrease in both accuracy and Gini score,to use 501 trees. 
    # Plotting the accuracy and Gini
    varImpPlot(rf, main="Mean Decrease in Accuracy and Gini for each variable")

![Accuracy and Gini scores](/AccNGini.png)
    
### Accuracy of Model 2 on training set and cross validation set
With the test data, accuracy obtained is 0.9997. 

    # results with training set
    pred2 <- predict(rf, newdata=trainset)
    confusionMatrix(pred2,trainset$classe)
    
    Confusion Matrix and Statistics

             # Reference
   # Prediction    A    B    C    D    E
            # A 3906    0    0    0    0
            # B    0 2658    0    0    0
            # C    0    0 2396    0    0
            # D    0    0    0 2252    0
            # E    0    0    0    0 2525
    
  # Overall Statistics
                                         
                  # Accuracy : 1          
                    # 95% CI : (0.9997, 1)
       # No Information Rate : 0.2843     
       # P-Value [Acc > NIR] : < 2.2e-16  
                                         
                     # Kappa : 1          
                     # Mcnemar's Test P-Value : NA         

   # Statistics by Class:

                       # Class: A Class: B Class: C Class: D Class: E
                       # Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
                       # Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
                       # Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
                       # Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
                       # Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
                       # Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
                       # Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
                       # Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

    # Results using validation test set
    predict2 <- predict(rf, newdata=testset)
    confusionMatrix(predict2,testset$classe)
    
    # Confusion Matrix and Statistics

             # Reference
   # Prediction    A    B    C    D    E
            # A 1671    5    0    0    0
            # B    2 1133    5    0    0
            # C    1    1 1020   11    1
            # D    0    0    1  952    0
            # E    0    0    0    1 1081

    # Overall Statistics
                                          
# Overall Statistics
                                          
              # Accuracy : 0.9952          
                # 95% CI : (0.9931, 0.9968)
   # No Information Rate : 0.2845          
   # P-Value [Acc > NIR] : < 2.2e-16       
                                          
                 # Kappa : 0.994           
                 # Mcnemar's Test P-Value : NA              

    # Statistics by Class:

                       # Class: A Class: B Class: C Class: D Class: E
                       # Sensitivity            0.9982   0.9947   0.9942   0.9876   0.9991
                       # Specificity            0.9988   0.9985   0.9971   0.9998   0.9998
                       # Pos Pred Value         0.9970   0.9939   0.9865   0.9990   0.9991
                       # Neg Pred Value         0.9993   0.9987   0.9988   0.9976   0.9998
                       # Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
                       # Detection Rate         0.2839   0.1925   0.1733   0.1618   0.1837
                       # Detection Prevalence   0.2848   0.1937   0.1757   0.1619   0.1839
                       # Balanced Accuracy      0.9985   0.9966   0.9956   0.9937   0.9994


### Conclusion
As the Random Forest model gave us the best result, use the Random Forest model on the validation test set to get the final results.

    # Predicting the class of the validation test set
    finres<-predict(rf,pretest)
    finres
    # The classes predicted for the 20 test cases was 
    # 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    #  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 

    # Writing the results to file
    pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
    }
    
    pml_write_files(finres)
