#Clear the environment
rm(list=ls())

#set the working directory
setwd(dir = "C:/Users/vinayak/Desktop/Car Fare prediction")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees',"usdm","randomForest","e1071","plyr", "dplyr","caTools",
      "tidyverse", "geosphere", "lubridate", "fpc")

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

################################## User defined Functions ################################################
divideDateTime <- function(data) {
  data = data %>%
    mutate(pickup_datetime = ymd_hms(pickup_datetime)
           ,year = year(pickup_datetime)
           ,month = month(pickup_datetime)
           ,day = day(pickup_datetime)
           ,dayOfWeek = wday(pickup_datetime)
           ,hour = hour(pickup_datetime))
  data=data[, !(colnames(data) %in% c("pickup_datetime"))]
  return(data)
}

convertIntoProperDataTypes <- function(data) {
  if(ncol(data)==11) {
    cnumber_factor = c(7,8,9,10,11)
    data$fare_amount = as.numeric(as.character(data$fare_amount))
   
  }
  else {
    cnumber_factor = c(6,7,8,9,10)
  }
  data[,cnumber_factor] <- lapply(data[,cnumber_factor] , factor)
  return(data)
}

getCnamesFactor <- function(data) {
  cnumber_factor = sapply(data, is.factor)
  factor_data = data[, cnumber_factor]
  return(colnames(factor_data))
}

getCnamesNumeric <- function(data) {
  cnumber_numeric = sapply(data, is.numeric)
  numeric_data = data[,cnumber_numeric]
  return(colnames(numeric_data))
}

getOptimalImputeMethod <- function(data) {
  actual = data[6,2]
  data[6,2] = NA
  
  dataKnn = data
  dataMean = data
  dataMedian = data
  
  #Mean Method
  dataMean$pickup_longitude[is.na(dataMean$pickup_longitude)] = mean(dataMean$pickup_longitude, na.rm = T)
  #Median Method
  dataMedian$pickup_longitude[is.na(dataMedian$pickup_longitude)] = median(dataMedian$pickup_longitude, na.rm = T)
  # kNN Imputation
  dataKnn = knnImputation(dataKnn, k = 3)
  
  result = c(actual, dataMean[6,2], dataMedian[6,2], dataKnn[6,2])
  names(result) = c("Actual", "Mean", "Median", "KNN")
  print(result)
}

imputeMissingValues <- function(data, cnames_numeric) {
  for(i in cnames_numeric) {
    data[,i][data[,i] %in% 0] = NA
  }
  print(paste("Before", sum(is.na(data))))
  # kNN Imputation
  data = knnImputation(data, k = 3)
  print(paste("After",sum(is.na(data))))
  return(data)
}

removeOutlier <- function(data, cnames_numeric) {
  for(i in cnames_numeric) {
    val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
    data[,i][data[,i] %in% val] = NA
  }
  sum(is.na(data))
  
  #Impute NA using KNN impute
  data = knnImputation(data, k = 3)
  return(data)
}

chiSquareTest <- function(data) {
  factor_data = train[, sapply(train, is.factor)]
  for (i in 1:length(cnames_factor)) {
    for(j in i+1:length(cnames_factor)) {
      if(j<=length(cnames_factor)) {
        print(paste(cnames_factor[i], " VS ", cnames_factor[j]))
        print(chisq.test(table(factor_data[,i],factor_data[,j])))
      }
    }
  }
}

featureScaling <- function(data, cnames_numeric) {
  for(i in cnames_numeric) {
    print(i)
    data[,i] = (data[,i] - mean(data[,i]))/sd(data[,i])
  }
  return(data)
}

################################## User defined Functions End ################################################

################################## Load dataset ################################################
train = read.csv("train_cab.csv")
test = read.csv("test.csv")

################################## Data Cleaning and preparation ################################################
train <- divideDateTime(train)
test <- divideDateTime(test)

train <- convertIntoProperDataTypes(train)
test <- convertIntoProperDataTypes(test)

str(train)

cnames_factor = getCnamesFactor(train)
cnames_numeric = getCnamesNumeric(train)

################################## Missing Value Analysis ################################################
getOptimalImputeMethod(train)

train = imputeMissingValues(train, cnames_numeric)

################################## Outlier Analysis ################################################
for (i in 1:length(cnames_numeric)) {
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames_numeric[i]), x = "fare_amount"), data = subset(train))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames_numeric[i],x="fare_amount")+
           ggtitle(paste("Box plot of fare_amount for",cnames_numeric[i])))
} 

gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,ncol=3)

train = removeOutlier(train, cnames_numeric)

################################## Analyse Data Distribution and Graphs ###########################################
ggplot(train, aes(x=pickup_longitude, y=pickup_latitude)) + geom_point()
ggplot(train, aes(x=dropoff_longitude, y=dropoff_latitude)) + geom_point()

cnames_numeric = cnames_numeric[-1]
#Analyze Distribution
for(i in 1:length(cnames_numeric)) {
  assign(paste0("gn",i), ggplot(data = train, aes_string(x = cnames_numeric[i])) + geom_histogram(bins = 25, fill="green", col="black")+ ggtitle(paste("Histogram of",cnames_numeric[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)

ggplot(data = train, aes(x=year, y= fare_amount)) + geom_bar(stat = 'identity')
ggplot(data = train, aes(x=month, y= fare_amount)) + geom_bar(stat = 'identity')
ggplot(data = train, aes(x=day, y= fare_amount)) + geom_bar(stat = 'identity') 
ggplot(data = train, aes(x=dayOfWeek, y= fare_amount)) + geom_bar(stat = 'identity') 
ggplot(data = train, aes(x=hour, y= fare_amount)) + geom_bar(stat = 'identity') 
ggplot(data = train, aes(x=hour, y= passenger_count)) + geom_bar(stat = 'identity') 
################################## Feature Selection ################################################
corrgram(train[getCnamesNumeric(train)], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#Chi-square test for correlation between categorical variable
chiSquareTest(data)

################################## Feature Scaling ################################################
qqnorm(train$pickup_longitude)
hist(train$dropoff_latitude)

train = featureScaling(train, cnames_numeric)
test = featureScaling(test, getCnamesNumeric(test))
################################## Model Development ################################################
#Clean the environment
rmExcept(c("train","test"))

#Split training data into training and test set
set.seed(123)
split = sample.split(train$fare_amount, SplitRatio = 0.8)
training_set = subset(train, split == TRUE)
test_set = subset(train, split == FALSE)

#Apply linear regression
regressor = lm(formula = fare_amount ~ ., data = training_set)
y_pred = predict(regressor, newdata = test_set[,-1])
sqrt(mean((test_set$fare_amount - y_pred)^2)/nrow(test_set))  #RMSE = 0.07478457

#Apply Decision tree
regressor = rpart(fare_amount ~ ., data = training_set, method = "anova")
y_pred = predict(regressor, newdata = test_set[,-1])
sqrt(mean((test_set$fare_amount - y_pred)^2)/nrow(test_set))  #RMSE = 0.0678244

#Apply Random Forest
set.seed(1234)
regressor = randomForest(x = training_set[,-1], y = training_set$fare_amount, ntree = 500)
y_pred = predict(regressor, newdata = test_set[,-1])
sqrt(mean((test_set$fare_amount - y_pred)^2)/nrow(test_set))  #RMSE = 0.05815893

#Random Forest :::: RMSE = 0.05815893 --> best among all
#applying test data
#testResult = predict(regressor, newdata = test)

#Note: RMSE may vary as training and test data may vary.