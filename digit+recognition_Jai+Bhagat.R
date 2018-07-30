############################ MNIST DATASET DIGIT Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
#  4.3 Polydot kernel
# 5 Hyperparameter tuning and cross validation
# 6 Conclusion

#####################################################################################

# 1. Business Understanding: 

#The objective is to identify each of a handwritten digits based on the 
#rectangular pixel displays as one of the 9 digits in the numeric system. 

#####################################################################################

# 2. Data Understanding: 
# Number of Instances: 70,000
# Number of Attributes: 785

#3. Data Preparation: 

##Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)
library(caTools)
# library(doParallel) 
# registerDoParallel()


#Setting up working directory 

setwd("C:\\Users\\sony\\Documents\\SVM")

#Loading the dataset 

digit <- read.csv("mnist_train.csv",stringsAsFactors = F,header = F)
digit_test <- read.csv("mnist_test.csv",stringsAsFactors = F,header = F)


dim(digit)  

#60000   785

dim(digit_test)

#10000   785

#Checking the structure of data

str(digit)
str(digit_test)

#Printing the first few rows

head(digit)
head(digit_test)

#Exploring the data

summary(digit)
summary(digit_test)

#Checking Blank Values in the data sets .

sapply(digit,function(x) length(which(x==" ")))
sapply(digit_test,function(x) length(which(x==" ")))

#As there are no blank values , we can proceed checking for NA values in the data set .

#Checking of NA values in the data sets. 

sapply(digit,function(x) sum(is.na(x)))
sapply(digit_test,function(x) sum(is.na(x)))

#There are no NA values as such in the dataset .

#Checking the duplicate rows in the data set .

unique(digit)
unique(digit_test)

#All unique rows .So contiuing the data preparation further . 

#Renaming the default 1st column as "number" in the two datasets  ,which contain the digits 0-9

colnames(digit)[1]<-c("number")
colnames(digit_test)[1]<-c("number")

#Converting the 1st column as factors since its the predicting variable .  

digit$number <-as.factor(digit$number)

digit_test$number <-as.factor(digit_test$number)

#str(digit)

#------------------- 4. MODEL BUILDING-------------------------

#Scaling the train and test data by dividing the every row pixel value by 255, except the first column. 

digit[, -1] <- digit[,-1]/255

digit_test[,-1]<-digit_test[,-1]/255

#RGB values are encoded as 8 bit integer [ranges from 0 to 256] . It's a industry standard to take
#0.0f as black and 1.0f as white . To Convert [0,255] to [0.0f,1.0f] we divide 255.

#Now
#Spliting the data and considering only 10 percent of the entire dataset for model building.
#To reduce the computation time.

set.seed(10)
indices <- sample.split(digit$number,SplitRatio = 0.10)
train = subset(digit, indices == TRUE)
test  = subset(digit, indices == FALSE)


#4.1: LINEAR MODEL SVM ,with kernel as vanilladot

Linear_model <- ksvm(number~.,data=train,scale = F ,kernel="vanilladot")

#Predicting the model results 

Eval_linear <- predict(Linear_model,digit_test)

# Confusion Matrix - Linear kernel

confusionMatrix(Eval_linear,digit_test$number)

#Accuracy : 0.9136

#4.2: Using RBF Kernel

Model_RBF <- ksvm(number~ ., data = train, scale = FALSE, kernel = "rbfdot")

#Predicting the model results
Eval_RBF<- predict(Model_RBF, digit_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,digit_test$number)

#Accuracy : 0.9537 


#4.3 Using Polydot

Model_Poly <- ksvm(number~.,data=train,scale = F,kernel="polydot")

#Predicting the model results
Eval_Poly<- predict(Model_Poly, digit_test)

#confusion matrix - Poly Kernel
confusionMatrix(Eval_Poly,digit_test$number)

#Accuracy : 0.9136


############   Hyperparameter tuning and Cross Validation #########################


trainControl <- trainControl(method="cv", number=5)

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 5 implies Number of folds in CV.


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid_linear <- expand.grid(.C=c(1,2,3,4,5))

grid_rbf <- expand.grid(.sigma=c(0.025, 0.05), .C=c(1,2,3,4,5) )

grid_Poly <- expand.grid(.degree = c(1,2), .scale = 1, .C =c(1,2,3,4) )

#CROSS VALIDATION FOR LINEAR MODEL

fit.svm_linear <- train(number~., data=train, method="svmLinear", metric=metric, 
                     tuneGrid=grid_linear, trControl=trainControl)

print(fit.svm_linear)

#Best Tune Value
# C = 1 Accuracy = 0.9059827  

plot(fit.svm_linear)

#Testing the accuracy of the cross validation model on the test data also .

Eval_CV_linear <- predict(fit.svm_linear,digit_test)

# Confusion Matrix 

confusionMatrix(Eval_CV_linear,digit_test$number)

#For C=1 Accuracy : 0.9136 , which is similar to the accuracy we got uusing the old model without optimization.


#CROSS VALIDATION FOR POLYDOT MODEL 

fit.svm_poly <- train(number~., data=train, method="svmPoly", metric=metric, 
                                                     tuneGrid=grid_Poly, trControl=trainControl)

print(fit.svm_poly)

#Best tune Value
#The final values used for the model were degree = 2, scale = 1 and C = 1.
#Accuracy 0.9524

plot(fit.svm_poly)

#Testing the accuracy of the cross validation model on the test data also .

Eval_CV_Poly <- predict(fit.svm_poly,digit_test)

# Confusion Matrix 

confusionMatrix(Eval_CV_Poly,digit_test$number)

#Accuracy : 0.9565 ,which is similar to the accuracy with bit improvement we got as using the old model without optimization.

#CROSS VALIDATION FOR RBF MODEL


fit.svm_rbf <- train(number~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid_rbf, trControl=trainControl)

print(fit.svm_rbf)

#Best Tune Value
#The final values used for the model were sigma = 0.025 and C = 4.
#Accuracy 0.9603

plot(fit.svm_rbf)

#Testing the accuracy of the cross validation model on the test data also .

Eval_CV_RBF<- predict(fit.svm_rbf,digit_test)

# Confusion Matrix 

confusionMatrix(Eval_CV_RBF,digit_test$number)

#Accuracy : 0.9636 , which is similar to the accuracy with bit improvement as  we got using the old model without optimization.

#----------------- 6. Conclusion-----------------------

#With a highest Accuracy of 96.36 the Radial Basis Function (RBF) kernel SVM , is the best fit model 
#The Accuracy was almost similar when we build the model using the training data set , which was 96.03
#And it performed well for the unseen test dataset(digit_test) also . 


#Intuitively, the Sigma parameter defines how far the influence of a single training example reaches,
#with low values meaning 'far' and high values meaning 'close'.
#The C parameter trades off misclassification of training examples against simplicity of the decision surface.
#A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly 
#by giving the model freedom to select more samples as support vectors.

#----------The best tune Values are Sigma =0.025 and CV = 4 .----------------------