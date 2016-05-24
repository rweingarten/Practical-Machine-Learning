# Practical Machine Learning Assignment
Rebecca Weingarten  
May 19, 2016  

##**Background**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement Â- a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this data set, the participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants toto predict the manner in which praticipants did the exercise.

The dependent variable or response is the "classe" variable in the training set.


##**Data**

Download and load the data:

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.5
```

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "U:/Documents/_rworking/pml-training.csv")
download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "U:/Documents/_rworking/pml-testing.csv")


trainingOrg = read.csv("U:/Documents/_rworking/pml-training.csv", na.strings=c("", "NA", "NULL"))
# data.train =  read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings=c("", "NA", "NULL"))

testingOrg = read.csv("U:/Documents/_rworking/pml-testing.csv", na.strings=c("", "NA", "NULL"))
dim(trainingOrg)
```

```
## [1] 19622   160
```

##**Pre-screening the data**

There are several approaches for reducing the number of predictors.

Remove variables that we believe have too many NA values.


```
## [1] 19622    60
```
Remove unrelevant variables There are some unrelevant variables that can be removed as they are unlikely to be related to dependent variable.

```
## [1] 19622    53
```
Check the variables that have extremely low variance (this method is useful nearZeroVar() )


```
## [1] 19622    53
```
Remove highly correlated variables 90% (using for example findCorrelation() )


```
## [1] 52 52
```

There are 52 variables. 

**Plot Variables** 
![](Practical_Machine_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

Next I remove the variables with high correlation: 

```
## Compare row 10  and column  1 with corr  0.992 
##   Means:  0.27 vs 0.168 so flagging column 10 
## Compare row 1  and column  9 with corr  0.925 
##   Means:  0.25 vs 0.164 so flagging column 1 
## Compare row 9  and column  4 with corr  0.928 
##   Means:  0.233 vs 0.161 so flagging column 9 
## Compare row 8  and column  2 with corr  0.966 
##   Means:  0.245 vs 0.157 so flagging column 8 
## Compare row 19  and column  18 with corr  0.918 
##   Means:  0.091 vs 0.158 so flagging column 18 
## Compare row 46  and column  31 with corr  0.914 
##   Means:  0.101 vs 0.161 so flagging column 31 
## Compare row 46  and column  33 with corr  0.933 
##   Means:  0.083 vs 0.164 so flagging column 33 
## All correlations <= 0.9
```

```
## [1] 19622    46
```

We get 19622 samples and 46 variables.

##**Split data to training and testing for cross validation.**

```
## [1] 13737    46
```

```
## [1] 5885   46
```

We got 13737 samples and 46 variables for training, 5885 samples and 46 variables for testing.

##**Analysis**
###***Regression Tree***

Now we fit a tree to these data, and summarize and plot it. First, we use the 'tree' package. It is much faster than 'caret' package.


```
## Warning: package 'tree' was built under R version 3.2.5
```

```
## 
## Classification tree:
## tree(formula = classe ~ ., data = training)
## Variables actually used in tree construction:
##  [1] "pitch_forearm"     "magnet_belt_y"     "accel_forearm_z"  
##  [4] "magnet_dumbbell_y" "roll_forearm"      "magnet_dumbbell_z"
##  [7] "accel_dumbbell_y"  "pitch_belt"        "yaw_belt"         
## [10] "accel_forearm_x"   "accel_dumbbell_z"  "gyros_belt_z"     
## Number of terminal nodes:  25 
## Residual mean deviance:  1.467 = 20120 / 13710 
## Misclassification error rate: 0.2812 = 3863 / 13737
```

![](Practical_Machine_files/figure-html/unnamed-chunk-9-1.png)<!-- -->


This tree has a ton of data and needs to be parsed down. First I am going to check the performance of the tree on the testing data by cross validation. 

##**Cross Validation**


```
## [1] 0.7119796
```

This rate of 0.6943076 is not very accurate.

###**Cross Validation** This shows a more narrow scope of the date: 



```
## $size
##  [1] 25 24 23 21 20 19 18 17 16 13 12  7  5  1
## 
## $dev
##  [1] 4155 4184 4370 4374 4538 4823 4967 5076 5139 6736 6768 6768 7268 9831
## 
## $k
##  [1]     -Inf  47.0000  96.0000  97.0000 125.0000 139.0000 149.0000
##  [8] 154.0000 168.0000 194.6667 202.0000 206.2000 255.0000 642.2500
## 
## $method
## [1] "misclass"
## 
## attr(,"class")
## [1] "prune"         "tree.sequence"
```

![](Practical_Machine_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

It shows that when the size of the tree goes down, the deviance goes up. It means the 21 is a good size (i.e. number of terminal nodes) for this tree. We do not need to prune it.

Suppose we prune it at size of nodes at 18.


```
## [1] 0.6542056
```

0.65 is slightly less than the previous determined error rate of 0.69. Therefore pruning did not hurt us with repect to misclassification errors, and gave us a simpler tree. We use less predictors to get almost the same result. By pruning, we got a shallower tree, which is easier to interpret.

The single tree is not good enough, so we are going to use bootstrap to improve the accuracy. We are going to try random forests.

##**Random Forests**

These methods use trees as building blocks to build more complex models. Random forests build lots of bushy trees, and then average them to reduce the variance.


```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, ntree = 100,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.66%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3899    4    1    1    1 0.001792115
## B   16 2633    9    0    0 0.009405568
## C    0   20 2373    3    0 0.009599332
## D    0    0   25 2226    1 0.011545293
## E    0    0    4    6 2515 0.003960396
```

![](Practical_Machine_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

we can see which variables have higher impact on the prediction.

##**Out-of Sample Accuracy**
Our Random Forest model shows OOB estimate of error rate: 0.72% for the training data. Now we will predict it for out-of sample accuracy.

Now lets evaluate this tree on the test data.


```
## [1] 0.9943925
```

0.99 means we got a very accurate estimate.

No. of variables tried at each split: 6. It means every time we only randomly use 6 predictors to grow the tree. Since p = 43, we can have it from 1 to 43, but it seems 6 is enough to get the good result.

##**Conclusion**
Now we can predict the testing data from the website.


```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

Those answers are going to submit to website for grading. It shows that this random forest model did a good job.
