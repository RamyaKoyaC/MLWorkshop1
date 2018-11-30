# MLWorkshop1

---
title: "Introduction to Hands-On Machine Learning"
author: "Ramya Koya"
date: "11/30/18"
output: ioslides_presentation
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE)
```

## Agenda 

- What is Machine Learning?
- Why do we need it?
- How to implement it? 
- Getting Hands-On ! 

## What is Machine Learning?

- A science that helps computers learn same like humans do.
-We learn from past experiences and when it comes to Machine it learns from data that are records of past experiences of an application domain.
- We feed in data to the machine, try to apply algorithms, train the machine on the fed data and now try working on the test data. 
- The traning data will help the machine learn and the test data concludes the learning capability of the model. 
- Once the machine produced good results, machine learning is finally acheived! 

## Types of Machine Learning 
- Supervised 
- Unsupervised
- Reinforced 

```{r, out.width = "600px"}
knitr::include_graphics("/storage/scratch2/rk0349/workshop1.png")
```
---
## Supervised ML
- In this, a function maps it's input to the output based on example pairs of input-ouput. 
- Be aware of the target function 
- Consider proper predictors that can define the target function

Few examples are, <br>
- Regression : To find a continous valued output (Stock Prices may vary in continuous values)<br>
- Classification: To find a discrete valued output(HIV Test, either positive or negative)

Our Hands-On Exercise has Linear Regression, Decision Trees, and Random Forests. 

## Unsupervised ML

In machine learning type, we are given datasets that have unlabeled responses and we now have to draw insights from this bundle of data. <br>

-It is vital to explore this type of data and try to figure out any kind of hidden patterns, groups, sets of data. 

-One of the famous unsupervised machine learning technique used is Clustering Analysis. 

## Reinforced ML
- In this, an agent will interact with it's environment and there will be certain rewards for each action performed. 
- state-action-reward
- The goal is to maximize the expected reward 
- The hand-heat-reward example  

# Getting Hands-On 
## Linear Regression 

A modelling method which is used to forecast a target value based on independent predictors. This helps in finding out the cause and effect relationship between the target and predictors. If the relationship is linear, then it is called a linear regression. <br>

In this example, we have considered the dataset of temperatures during World War 2. <br>
Target - Mean Temperature <br>
Predictor - Min Temperature

Here we are fitting a linear regression model: 

```{r temperature, echo = TRUE}
temperature <- read.csv("/storage/scratch2/rk0349/Summary of Weather.csv")
fit <- lm(temperature$MeanTemp ~ temperature$MinTemp)
```
Target = (Slope * Predictor) + Intercept 
---
```{r temperature1, echo =TRUE}
summary(fit)
```

Now we make predictions of other possible values of the target variable using the linear regression equation
---
```{r temperature2, echo=TRUE}
predictions <- predict(fit,temperature)
head(predictions)
```

The mean square error 
---
```{r temperature5, echo=TRUE}
mse <- mean((temperature$MeanTemp - predictions)^2)
print(mse)
```

Plotting the Predictor v/s Target
---
```{r temperature3, echo=TRUE}
plot(temperature$MinTemp, temperature$MeanTemp)
```

Plotting the obtained predictions with the target 
---
```{r temperature4, echo=TRUE}
plot(predictions, temperature$MeanTemp)
```

## Decision Trees 
-One of famous machine learning techniques used for decision making is decision trees. 
-Considering the conditions, decisions are taken and branches are drawn. 
-It is upside down as the root is the beginning of the tree-like structure. 


```{r, out.width = "300px"}
knitr::include_graphics("/storage/scratch2/rk0349/decision1.png")
```


In this breast cancer dataset we have 1 million instances and 15 columns that give information.
```{r dt, echo=TRUE}
# Decision Tree 
cancer_data <- read.csv("/storage/scratch2/share/DSA/clean_BayesianNetworkGenerator_breast-cancer_small.csv")
cancer_data<-cancer_data[1:100,] #selecting only the 100 rows 
```



```{r dt1}
cancer_data$age <- factor(cancer_data$age) # Converting the dataset in categorical data
is.factor(cancer_data[,1])
cancer_data$menopause <- factor(cancer_data$menopause)
is.factor(cancer_data[,2])
cancer_data$tumor.size <- factor(cancer_data$tumor.size)
is.factor(cancer_data[,3])
cancer_data$inv.nodes <- factor(cancer_data$inv.nodes)
is.factor(cancer_data[,4])
cancer_data$node.caps <- factor(cancer_data$node.caps)
is.factor(cancer_data[,5])
cancer_data$deg.malig <- factor(cancer_data$deg.malig)
is.factor(cancer_data[,6])
cancer_data$breast <- factor(cancer_data$breast)
is.factor(cancer_data[,7])
cancer_data$breast.quad <- factor(cancer_data$breast.quad)
is.factor(cancer_data[,8])
cancer_data$irradiat <- factor(cancer_data$irradiat)
is.factor(cancer_data[,9])
cancer_data$Class <- factor(cancer_data$Class)
is.factor(cancer_data[,10]) 
```


To see the overall summary of the dataset
---
```{r dt2,echo=TRUE}
summary(cancer_data)# printing out the summary
```

```{r dt7, echo=TRUE}
head(cancer_data)
```

Splitting the data into Train and test
---
```{r dt3, echo=TRUE}
set.seed(100) 
train <- sample(nrow(cancer_data), 0.75*nrow(cancer_data)) #splitting the data in train and test sets
train_set <- cancer_data[train,]
test_set <- cancer_data[-train,]
```

Train_set
---
```{r dt4, echo=FALSE}
summary(train_set)
```


Test_set
---
```{r dt5, echo=FALSE}
summary(test_set) #viewing the models test set
```


```{r dt8, echo=FALSE}
library(partykit)
library(rpart)
cols <- c('Class', 'age', 'tumor.size')
cancer_data[cols] <- lapply(cancer_data[cols], as.factor)
set.seed(1)
train <- sample(1:nrow(cancer_data), 0.75 * nrow(cancer_data))
```

Building a decision tree model, taking age as a target. The train data is 75% and the test data is 25%
---
```{r dt9, echo=TRUE}
cancerTree <- rpart(age ~ ., data = cancer_data[train, ], method = 'class')
```

Decision Tree
---
```{r dt11, echo=TRUE}
plot(as.party(cancerTree))
```

##Random Forests

Loading the randomForest library 
---
```{r rf1, echo=TRUE}
library(randomForest)
```

---
```{r rf2, echo=FALSE}
cancer_data <- read.csv("/storage/scratch2/share/DSA/clean_BayesianNetworkGenerator_breast-cancer_small.csv")
cancer_data<-cancer_data[1:500,] #selecting only the 500 rows 
cancer_data$age <- factor(cancer_data$age)
is.factor(cancer_data[,1])
cancer_data$menopause <- factor(cancer_data$menopause)
is.factor(cancer_data[,2])
cancer_data$tumor.size <- factor(cancer_data$tumor.size)
is.factor(cancer_data[,3])
cancer_data$inv.nodes <- factor(cancer_data$inv.nodes)
is.factor(cancer_data[,4])
cancer_data$node.caps <- factor(cancer_data$node.caps)
is.factor(cancer_data[,5])
cancer_data$deg.malig <- factor(cancer_data$deg.malig)
is.factor(cancer_data[,6])
cancer_data$breast <- factor(cancer_data$breast)
is.factor(cancer_data[,7])
cancer_data$breast.quad <- factor(cancer_data$breast.quad)
is.factor(cancer_data[,8])
cancer_data$irradiat <- factor(cancer_data$irradiat)
is.factor(cancer_data[,9])
cancer_data$Class <- factor(cancer_data$Class)
is.factor(cancer_data[,10])
```

Splitting data and building model 
---
```{r rf3, echo=TRUE}
set.seed(100) 
train <- sample(nrow(cancer_data), 0.7*nrow(cancer_data)) #splitting the data in train and test sets
train_set <- cancer_data[train,]
test_set <- cancer_data[-train,]
model1 <- randomForest(Class ~ ., data = train_set, ntree = 500, mtry = 6, importance = TRUE)  #building a model 2 alongside the confusion matrix
model1 #viweing the built model 
```


```{r rf7, echo=TRUE}
plot(model1) #plotting the number of trees v/s error
```

Working with the predicted values
---
```{r rf4, echo=TRUE}
pred_train <- predict(model1, train_set, type = "class") 
mean(pred_train == train_set$Class)  #predicting the train set on model 1
table(pred_train, train_set$Class)  
pred_test <- predict(model1, test_set, type="class")
mean(pred_test == test_set$Class) #preciting the test set on model 1
table(pred_test, test_set$Class)
```

To find the variable importances 
---
```{r rf5, echo=TRUE}
importance(model1) # finding the Importances for model 1
```

Variable Importance Plot 
---
```{r rf8, echo=TRUE}
varImpPlot(model1)# Variable important plot for model 1
```

To find the accuracy with the mtry
---
```{r rf6, echo=TRUE}
a=c()
i=5
for (i in 3:8) {
  model3 <- randomForest(Class ~ ., data = train_set, ntree = 500, mtry = i, importance = TRUE)
  pred_test <- predict(model3, test_set, type = "class")
  a[i-2] = mean(pred_test == test_set$Class)
}
a
```

Plotting accuracy v/s mtry
---
```{r rf10, echo=TRUE}
plot(3:8,a)
```

IML


About the dataset:
Housing Values in Suburbs of Boston
Description
The Boston data frame has 506 rows and 14 columns

crim
per capita crime rate by town.

zn
proportion of residential land zoned for lots over 25,000 sq.ft.

indus
proportion of non-retail business acres per town.

chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox
nitrogen oxides concentration (parts per 10 million).

rm
average number of rooms per dwelling.

age
proportion of owner-occupied units built prior to 1940.

dis
weighted mean of distances to five Boston employment centres.

rad
index of accessibility to radial highways.

tax
full-value property-tax rate per \$10,000.

ptratio
pupil-teacher ratio by town.

black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat
lower status of the population (percent).

medv
median value of owner-occupied homes in \$1000s.

---
```{r iml, echo=TRUE}
library("mlr")
data("Boston", package  = "MASS")

# create an mlr task and model
tsk = makeRegrTask(data = Boston, target = "medv")
lrn = makeLearner("regr.randomForest", ntree = 100)
mod = train(lrn, tsk)
library("iml")
X = Boston[which(names(Boston) != "medv")]
predictor = Predictor$new(mod, data = X, y = Boston$medv)
imp = FeatureImp$new(predictor, loss = "mae")
```

Variable Importance Plot
---
```{r iml1, echo=TRUE}
plot(imp)
```

Plot with the most important feature and the predictor 
---
```{r iml2, echo=TRUE}
pdp.obj = Partial$new(predictor, feature = "lstat")
plot(pdp.obj)
```

Growing a tree with considering 2 feature
---
```{r iml3, echo=TRUE}
tree = TreeSurrogate$new(predictor, maxdepth = 2)
plot(tree)
```


---
## Thank you for your time! 
-Reach out to us at email: hpc-admin@unt.edu
-Find the code on: Github 
