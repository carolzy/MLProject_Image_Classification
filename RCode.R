####################################################################
######### STAT 154 FINAL PROJECT ###################################
######### JONATHAN WANG, ALEX DOMBROWSKI, Yue (Carol) Zhu ##########

# Set appropriate directory and load data
      setwd("~/Documents/CalSpring2014/Stat154/Project/")
      train = read.csv('train.csv', header=F)
      test = read.csv('test.csv', header=F)
# Sanity Check 
dim(train)  
  # [1] 800 513
dim(test)
  # [1] 1888  512
names(test) = names(train)[2:513]  # The predict function wants the column names
                                    # of the data frames to be the same ... fine.
names(train)[1] = 'Response'
train$Response = as.factor(train$Response)

#######################
##   List of Methods ##
#######################
  # knn
  # logistic
  # sparse logistic (glmnet)
  # lda / qda / RDA
  # gams (library "gam" or "mgcv") : generalized additive model
  # CART / Bagging / Random Forests
  # SVM


# Before submitting predictions to Kaggle, use the training set to get an idea of 
  # what prediction accuracy is. Split the training into a `subtest' and `subtrain' set. 
train_indices = sample(seq(1,nrow(train)), size=250, replace=FALSE)   
    # I use 250 of the 800 observations in the training set to form the subtrain set
    # because 250/800 is approx the same as nrow(train)/(nrow(train)+nrow(test))
test_indices = (1:800)[-train_indices]
subtrain = train[train_indices,]
subtest = train[test_indices,]

names(subtrain)[1] = 'Response'
subtrain$Response = as.factor(subtrain$Response)


###############################################
########    METHOD: knn()    ##################
###############################################
# Classification Results: ............66% accurate according to kaggle
#                                     72% when splitting true training into test and train                                                               
# Strategy...............................Find optimal k using CV, then use knn. 
#                                      Basically what Derek did but first CV to find k
# Concerns: Curse of dimensionality

install.packages('class')
library(class)

# Does data need to be centered and scaled?
means = apply(train[,-1],2, mean)
hist(means)
sds = apply(train[,-1],2, sd)
hist(sds)
  # Running on the SPLIT TRAINING DATA without scaling gave k=6 and accuracy of 71%
  # Scaling gave k=4 and accuracy of 72%

# Use CV to find the optimal k. 
knnCVplot = function(Train, kmin, kmax, scale_data=TRUE){
  success_rates = c()
  for (K in kmin:kmax){  
    if (scale_data){
      LOOCV_result = knn.cv(train=scale(Train[,-1]), cl=as.factor(Train[,1]), k=K) 
    } else {
      LOOCV_result = knn.cv(train=Train[,-1], cl=as.factor(Train[,1]), k=K) 
    }
    success_rate = sum(LOOCV_result == Train[,1]) / nrow(Train)
    success_rates = c(success_rates, success_rate)
    }
  plot(x=seq(kmin,kmax), y=success_rates, main='Success Rates from knn CV', 
       ylab='Prediction Accuracy', xlab='k')
}

knnCVplot(Train=train, 1,20) # *This takes about 2min to run.
  # k=4 looks like a solid choice from my simulation 

# Can run on training set split into two to get ball park idea of what kaggle will give:
myKNNsim = function() {
  indices = sample(1:nrow(train), size=400, replace=FALSE)
  indices2 = c(1:800)[-indices]
  subtest = train[sort(indices),]
  subtrain = train[indices2,]
  res = knn(train=scale(subtrain[,-1]), test=scale(subtest[,-1]), cl=as.factor(subtrain[,1]), k=4)
  success_rate = sum(res == subtest[,1]) / nrow(subtest)
  return(success_rate)
}
estimated_accuracy_rates = replicate(n=25, expr=myKNNsim())
hist(estimated_accuracy_rates)
summary(estimated_accuracy_rates)

# Run knn with the optimally chosen k from knnCVplot()
predictions = knn(train=scale(train[,-1]), test=scale(test), cl=as.factor(train[,1]), k=4)
as.integer(predictions)
submissions = cbind((1:nrow(test)), predictions)
write.table(submissions, file = "SubmitMe.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)

###############################################
########    METHOD: knn + PCA    ##################
###############################################
# Classification Results: ............                                   
# Strategy...............................
# Concerns: 






###############################################
########    METHOD: Bagging    ##################
###############################################
# Classification Results: ............0.64301 accuracte according to kaggle
#                                     0.74 when using training set only 
# Strategy...........................
# Concerns...........................

install.packages('ipred')
library(ipred)

# Create Tree
baggingTree1 = bagging(Response~., data=subtrain, nbagg=100)
# Make predictions on subtest data
baggingTree1.predictions = predict(baggingTree1, newdata=subtest[,-1])
# Prediction accuracy
sum(as.integer(baggingTree1.predictions)==subtest[,1])/nrow(subtest)
# This gives ~0.74 accuracy.

# Now implement on actual test data 
baggingTree2 = bagging(Response~., data=train, nbagg=100)
baggingTree2.predictions = predict(baggingTree2, newdata=test)


# Put the predictions in a csv file to be submitted to kaggle.
submissions = cbind((1:nrow(test)), as.integer(baggingTree2.predictions))
write.table(submissions, file = "Bagging.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)


###############################################
########    METHOD: Random Forest    ##########
###############################################
# Classification Results: ............ 0.71081 accurate according to Kaggle
#                          ............0.75 using training
# Strategy...............................
# Concerns

install.packages('randomForest')
library(randomForest)
RandomForestModel1 = randomForest(subtrain[,-1],
                                 subtrain[,1], ntrees=1000)
RandomForest.prediction1 = predict(RandomForestModel1, 
                                           newdata=subtest[,-1])
sum(as.integer(RandomForest.prediction1) == subtest[,1])/nrow(subtest)
# [1] 0.75

# Now on actual test data
RandomForestModel2 = randomForest(train[,-1],
                                  train[,1], ntrees=1000)
RandomForest.prediction2 = predict(RandomForestModel2, 
                                   newdata=test)

submissions = cbind((1:nrow(test)), as.integer(RandomForest.prediction2))
write.table(submissions, file = "RandomForest.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)

# What proportion of predictions do bagging and random forests agree on?
sum(as.integer(baggingTree2.predictions) == as.integer(RandomForest.prediction2)) / nrow(test)
# 0.7944915


###############################################
########    METHOD: Boosting   ######
###############################################
# Classification Results: ............ 0.75847 according to kaggle
#                          ............0.78 when using training set for train and test
# Strategy...............................
# Concerns...............................I didn't tune interaction.depth or shrinkage yet


## ...http://cran.r-project.org/web/packages/gbm/gbm.pdf
## https://www.youtube.com/watch?v=L6BlpGnCYVg     good explanation of multiclass ada by the inventor 

install.packages('gbm')
library(gbm)
install.packages('nnet')   # I want to use which.is.max() to return index of maximal value in a vector
library(nnet)

BoostingModel1 = gbm(Response~., data=subtrain, distribution='multinomial',  # multinomial is default when response is a factor
                     interaction.depth=2, n.trees=1500, shrinkage=0.01,
                     verbose=FALSE)
BoostingPreds = predict(BoostingModel1, newdata=subtest[,-1], n.trees=1500, type='response')

# The output, preds, is a dim(subtest) by 8 matrix with (i,j) entry being the 
# predicted probability test observation i is in class j.
# Convert probability to class
probs_to_classes = function(obj){
  BoostingPreds_classes = c()
  for (i in 1:nrow(obj)){
    BoostingPreds_classes[i] = which.is.max(obj[i,,1])
  }
  return(BoostingPreds_classes)
}
BoostingPreds_classes1 = probs_to_classes(BoostingPreds)

# Check the accuracy:
sum(BoostingPreds_classes1==as.numeric(subtest[,1])) / nrow(subtest)

# Now on the test data:
BoostingModel2 = gbm(Response~., data=train, distribution='multinomial',  # multinomial is default when response is a factor
                     interaction.depth=2, n.trees=1500, shrinkage=0.01,
                     verbose=FALSE)
BoostingPreds2 = predict(BoostingModel2, newdata=test, n.trees=1500, type='response')
BoostingPreds_classes2 = probs_to_classes(BoostingPreds2)

submissions = cbind((1:nrow(test)), as.integer(BoostingPreds_classes2))
write.table(submissions, file = "Boosting.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)



###############################################
########    METHOD: SVM   ######
###############################################
# Classification Results: .............0.79025 according to kaggle
#                          ............# 0.8072727 using subtest/subtrain
# Strategy...............................
# Concerns............................216 support vectors when model is trained
#                                 on subtrain, which has only 250 observations
#                                 is high, but p.422 in book had 85%

install.packages('e1071')
library(e1071)

# Find good choice of C, gamma. Runnning this took 2hr.
tuning_params = tune.svm(x=subtrain[,-1], y=subtrain[,1], cost=2^(-5:10), gamma=2^(-10:5))
summary(tuning_params)

# Run model with above parameters
SVMmodel1 = svm(Response~., data=subtrain, type='C-classification', 
                cost=2, gamma=0.001953125)
summary(SVMmodel1)

# Make predictions on subtest set
SVMpredictions = predict(SVMmodel1, newdata=subtest[,-1])
sum(as.numeric(SVMpredictions) == as.numeric(subtest[,1])) / nrow(subtest)
# 0.8072727

# Now on true test set
# Find good choice of C, gamma. Runnning this took 2hr.
tuning_params2 = tune.svm(x=train[,-1], y=train[,1], cost=2^(-5:10), gamma=2^(-10:5))
summary(tuning_params2)

# Run model with above parameters
SVMmodel2 = svm(Response~., data=train, type='C-classification', 
                cost=8, gamma=0.0009765625)
summary(SVMmodel2)
# 544 support vectors out of 800 observations

# Make predictions
SVMpredictions2 = predict(SVMmodel2, newdata=test)

submissions = cbind((1:nrow(test)), as.integer(SVMpredictions2))
write.table(submissions, file = "SVM.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)

## More SVM with different kernels 
# Linear Kernel: # [1] 0.8163636
SVMmodel3 = svm(Response~., data=subtrain, type='C-classification', method='linear',
                cost=2, gamma=0.001953125)
summary(SVMmodel3)
SVMpredictions3 = predict(SVMmodel3, newdata=subtest[,-1])
sum(as.numeric(SVMpredictions3) == as.numeric(subtest[,1])) / nrow(subtest)

# Polynomial Kernel: # [1] 0.8163636
SVMmodel4 = svm(Response~., data=subtrain, type='C-classification', method='polynomial',
                cost=2, gamma=0.001953125)
summary(SVMmodel4)
SVMpredictions4 = predict(SVMmodel4, newdata=subtest[,-1])
sum(as.numeric(SVMpredictions4) == as.numeric(subtest[,1])) / nrow(subtest)

# Sigmoid Kernel: # [1] 0.8163636
SVMmodel5 = svm(Response~., data=subtrain, type='C-classification', method='sigmoid',
                cost=2, gamma=0.001953125)
summary(SVMmodel5)
SVMpredictions5 = predict(SVMmodel5, newdata=subtest[,-1])
sum(as.numeric(SVMpredictions5) == as.numeric(subtest[,1])) / nrow(subtest)

# Kernel other than radial basis. On the whole train set. 
SVMmodel6 = svm(Response~., data=train, type='C-classification', method='sigmoid',
                cost=8, gamma=0.0009765625)
summary(SVMmodel6)
# 544 support vectors out of 800 observations

# Make predictions
SVMpredictions6 = predict(SVMmodel6, newdata=test)
submissions = cbind((1:nrow(test)), as.integer(SVMpredictions6))
write.table(submissions, file = "SVMsigKernel.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)



###############################################
########    METHOD: OVA SVM   ######
###############################################
# Classification Results: ............. according to kaggle
#                          ............ 0.8054545 using subtest/subtrain
# Strategy...............................Use OVA from Rifkin/Klautau paper
# Concerns.............................


# Need function to change everything but the response value 'one' to something else
changeto9 = function(vector, k){
  # Helper function for OVA()
  # changeto9(c(1,2,3,4,4),3)
  # [1] 9 9 3 9 9
  # Levels: 3 9
  vector = as.numeric(vector)
  for (i in 1:length(vector)){
    if (vector[i]!=k){
      vector[i] = 9
    }
  }
  return(as.factor(vector))
}

# Here's how this works when the 'One' in 'One vs All' is 1:
subtrain1 = subtrain
subtrain1[,1] = changeto9(subtrain1[,1], 1)
subtest1 = subtest
subtest1[,1] = changeto9(subtest1[,1], 1)
SVMmodel01 = svm(Response~., data=subtrain1, type='C-classification', 
                 cost=2, gamma=0.001953125)
summary(SVMmodel01)
SVMpredictions01 = predict(SVMmodel01, newdata=subtest1[,-1])
sum(as.numeric(SVMpredictions01) == as.numeric(subtest1[,1])) / nrow(subtest)
# [1] 0.9363636
# Not bad, but the error will likely build because we have to do this for each class

OVA = function(one, training, testing, Cost, Gamma, Full=False){
  # This function build a one verse all SVM and outputs a vector of predictions.
  training[,1] = changeto9(training[,1], one)
  SVMmodel = svm(Response~., data=training, type='C-classification', 
                 cost=Cost, gamma=Gamma)
  if (Full){
    SVMpredictions = predict(SVMmodel, newdata=testing)
  } else {
    SVMpredictions = predict(SVMmodel, newdata=testing[,-1]) 
  }
  return(as.integer(as.character(SVMpredictions)))
}

prediction_matrix = cbind(OVA(1, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(2, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(3, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(4, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(5, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(6, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(7, subtrain, subtest, 2, 0.001953125, F), 
                          OVA(8, subtrain, subtest, 2, 0.001953125, F))
s = c()
for (i in 1:nrow(prediction_matrix)){
  if (length(unique(prediction_matrix[i,])) != 2){
    s = c(s, i)
  }
}
# 28% of the observations couldn't be predicted by OVA SVM. It either couldn't classify
# them (row is all 9s) or were classified to more than 1 class. 

# This is the predictions on subtest using the regular multiclass SVM on obs that OVA couldnt classify
SVMmodelog = svm(Response~., data=subtrain, type='C-classification', 
                 cost=2, gamma=0.001953125)
SVMpredictions = predict(SVMmodel1, newdata=subtest[,-1])
SVMog = as.numeric(as.character(SVMpredictions))
final_preds = c()
for (i in 1:nrow(prediction_matrix)){
  if (length(unique(prediction_matrix[i,]))==2){
    prediction = unique(prediction_matrix[i,])[which(unique(prediction_matrix[i,])<9)]
    final_preds = c(final_preds, prediction)
  } else {
    final_preds = c(final_preds, SVMog[i])
  }
}

sum(final_preds==as.numeric(subtest[,1]))/nrow(subtest)
# [1] 0.8054545
# Conclusion: When splitting the training into subtest and subtrain, it doesn't seem that 
#             OVA will improve multiclass SVM. 

### Now samething but on the whole training set:
prediction_matrix_full = cbind(OVA(1, train, test, 8, 0.0009765625, T),
                               OVA(2, train, test, 8, 0.0009765625, T),
                               OVA(3, train, test, 8, 0.0009765625, T),
                               OVA(4, train, test, 8, 0.0009765625, T),
                               OVA(5, train, test, 8, 0.0009765625, T),
                               OVA(6, train, test, 8, 0.0009765625, T),
                               OVA(7, train, test, 8, 0.0009765625, T),
                               OVA(8, train, test, 8, 0.0009765625, T))
SVMmodelFULLog = svm(Response~., data=train, type='C-classification', 
                     cost=8, gamma=0.0009765625)
SVMpredictionsFULL = predict(SVMmodelFULLog, newdata=test)
SVMogFULL = as.numeric(as.character(SVMpredictionsFULL))
final_preds_FULL = c()
for (i in 1:nrow(prediction_matrix_full)){
  if (length(unique(prediction_matrix_full[i,]))==2){
    prediction = unique(prediction_matrix_full[i,])[which(unique(prediction_matrix_full[i,])<9)]
    final_preds_FULL = c(final_preds_FULL, prediction)
  } else {
    final_preds_FULL = c(final_preds_FULL, SVMogFULL[i])
  }
}                  

submissions = cbind((1:nrow(test)), as.integer(final_preds_FULL))
write.table(submissions, file = "SVMova.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)



########
# Big picture idea for improving prediction:
# Compare the predictions of all methods. For each observation in the test set, take the 
# majority vote of the methods to get perhaps a more accurate predition.
# See which testing obervations are difficult to classifty (e.g. methods are very 
# split on which class to place it) and focus on those. Maybe see which training observations
# are similar to those test observation and see how the methods did on those training obs. 
#######


# Try the above aggregation suggestion with Boosting, Random Forests, and SVM
# According to kaggle, this gives a 0.78496 which sucks...surprised it did worse than SVM alone
Boost = read.csv('Boosting.csv')
RF = read.csv('RandomForest.csv')
SVM = read.csv('SVM.csv')
all = cbind(Boost$Predictions, RF$Predictions, SVM$Predictions)

most_often = function(vector){
  # Helper function. Returns the most often occurring element in a vector of length 3, assuming no ties.
  # > most_often(c(6,4,6))
  # [1] 6
  if (length(unique(vector))==1){
    return(vector[1])
  } else {
    if (sum( unique(vector)[1]==vector) ==2){
      return(unique(vector)[1])
    } else {
      return(unique(vector)[2])
    }
  }
}

worst = c()
for (i in 1:nrow(all)){
  if (length(unique(all[i,]))==3){
    worst = c(worst, i)
  }
}

combined = c()
for (i in 1:nrow(all)){
  if (i %in% worst){
    combined = c(combined, all[i,3])
  } else {
    combined = c(combined, most_often(all[i,]))
  }
}

submissions = cbind((1:nrow(test)), as.integer(combined))
write.table(submissions, file = "BoostRFSVMcombined.csv", sep = ",", col.names = c("Id", "Predictions"), row.names = F)
