# Load the dataset
data<-read.csv("C:\\Users\\91838\\Downloads\\Boston University\\CS699 - Data Mining\\Sarthak_Pattnaik-Aditya_Balasubramanian-Project_Proposal\\recedivist_dataset.csv")

# Remove NA values from the data
data <- na.omit(data)

# Calculate the interquartile range (IQR) for each column
q1 <- apply(data, 2, quantile, probs = 0.25)
q3 <- apply(data, 2, quantile, probs = 0.75)
iqr <- q3 - q1


# Identify outliers using Tukey's method
outliers <- apply(data, 2, function(x) {
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  x[x < lower | x > upper] <- NA
  return(x)
})

# Remove rows with missing values
dataset <- na.omit(outliers)
dataset <- data.frame(dataset)
write.csv(dataset, file = "recidivist_datatset_preprocessed.csv")

#data exploration
head(dataset, n=10)


ggplot(dataset, aes(x = age)) +
  geom_histogram(fill = "steelblue", color = "white") +
  labs(title = "Age Distribution", x = "Age", y = "Frequency")

# Create a scatter plot of age vs. priors_count, colored by race
ggplot(dataset, aes(x = age, y = priors_count, color = race)) +
  geom_point(alpha = 0.5) +
  labs(title = "Age vs. Priors Count by Race", x = "Age", y = "Priors Count")

# Create a bar chart of score_text, grouped by sex
ggplot(dataset, aes(x = score_text, fill = sex)) +
  geom_bar(position = "dodge") +
  labs(title = "Score Text by Sex", x = "Score Text", y = "Count")

# Boxplot of Decile Score by Race
boxplot(data$decile_score.1 ~ data$race, main="Boxplot of Decile Score by Race", xlab="Race", ylab="Decile Score")
boxplot(dataset$decile_score.1 ~ dataset$race, main="Boxplot of Decile Score by Race", xlab="Race", ylab="Decile Score")


library(randomForest)
library(caret)
library(rsample)
library(RWeka)
library(rpart)
library(MASS)
library(kernlab)
library(pROC)
library(tidyverse)


dataset$is_recid <- as.factor(dataset$is_recid)

#Weka
set.seed(123)
trainIndex <- createDataPartition(dataset$is_recid, p = .8, list = FALSE)
train_data <- dataset[ trainIndex,]
test_data <- dataset[-trainIndex,]
ctrl <- trainControl(method="cv", number=10)
model <- train(is_recid ~ ., data=train_data, method="J48", trControl=ctrl)
predictions <- predict(model, newdata=test_data)
conf_matrix_Weka <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_Weka$byClass[2]
fp_rate <- conf_matrix_Weka$byClass[3]

precision <- conf_matrix_Weka$byClass[1]
recall <- conf_matrix_Weka$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)


#NB
# Load required libraries
library(e1071)

# Define the control for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
nb_model <- train(is_recid ~ ., data = train_data, method = "nb",
                  trControl = train_control)

predictions <- predict(nb_model, newdata=test_data)
conf_matrix_NB <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NB$byClass[2]
fp_rate <- conf_matrix_NB$byClass[3]

precision <- conf_matrix_NB$byClass[1]
recall <- conf_matrix_NB$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_NB <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
             f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_NB)


#Random Forest
# Define the control for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
set.seed(123)
rf_model <- train(is_recid ~ ., data = train_data, method = "rf",
                  trControl = train_control)

predictions <- predict(rf_model, newdata=test_data)
conf_matrix_rf <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_rf$byClass[2]
fp_rate <- conf_matrix_rf$byClass[3]

precision <- conf_matrix_rf$byClass[1]
recall <- conf_matrix_rf$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_rf <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
             f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_rf)

#SVM
svmFit <- train(is_recid ~ ., data = train_data, method = "svmRadial",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10, 100)))

testPred <- predict(svmFit, newdata = test_data)
conf_matrix_SVM <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_SVM$byClass[2]
fp_rate <- conf_matrix_SVM$byClass[3]

precision <- conf_matrix_SVM$byClass[1]
recall <- conf_matrix_SVM$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_SVM <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
              f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_SVM)

#dataset <- scale(dataset[, -which(names(dataset) == "v_type_of_assessment")])

# Neural Net model

#setting up cross validation method
cv <- trainControl(method = "cv", number = 10)

#define the parameter grid to search over
grid <- expand.grid(size = 1:10, decay = c(0, 0.1, 1, 2))

nn_model <- train(is_recid ~ ., data = train_data, method = "nnet", tuneGrid = grid, trace=FALSE, maxit=100, MaxNWts = 1000, trControl = cv)

predictions <- predict(nn_model, test_data)

conf_matrix_NN <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NN$byClass[2]
fp_rate <- conf_matrix_NN$byClass[3]

precision <- conf_matrix_NN$byClass[1]
recall <- conf_matrix_NN$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_nn <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
             f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_nn)

#split the dataset into training and test set using 66% training
set.seed(31)
split <- initial_split(dataset, prop = 0.66, strata = is_recid)
train <- training(split)
test <- testing(split)

train$is_recid <- as.numeric(train$is_recid)

###################Correlation based feature selection################
# Compute the correlation matrix
correlation_matrix <- cor(train)

# Extract the correlations with is_recid
correlations <- correlation_matrix[, "is_recid"]

# Sort the correlations in descending order
sorted_correlations <- sort(correlations, decreasing = TRUE)

# Print the sorted correlations
sorted_correlations <- sorted_correlations[-1]
print(sorted_correlations[1:4])

dataset1 <- dataset[c('decile_score.1', 'v_decile_score', 'priors_count', 'juv_misd_count', 'is_recid')]
dataset1$is_recid <- as.factor(dataset1$is_recid)

#Weka
set.seed(123)
trainIndex <- createDataPartition(dataset1$is_recid, p = .8, list = FALSE)
train_data <- dataset1[ trainIndex,]
test_data <- dataset1[-trainIndex,]
ctrl <- trainControl(method="cv", number=10)
model <- train(is_recid ~ ., data=train_data, method="J48", trControl=ctrl)
predictions <- predict(model, newdata=test_data)
conf_matrix_Weka_corr <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_Weka_corr$byClass[2]
fp_rate <- conf_matrix_Weka_corr$byClass[3]

precision <- conf_matrix_Weka_corr$byClass[1]
recall <- conf_matrix_Weka_corr$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_weka_corr <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                    f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_weka_corr)

#Naive Bayes
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
nb_model <- train(is_recid ~ ., data = train_data, method = "nb",
                  trControl = train_control)

predictions <- predict(nb_model, newdata=test_data)
conf_matrix_NB_corr <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NB_corr$byClass[2]
fp_rate <- conf_matrix_NB_corr$byClass[3]

precision <- conf_matrix_NB_corr$byClass[1]
recall <- conf_matrix_NB_corr$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_NB_corr <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_NB_corr)

#Random Forest
train_control <- trainControl(method = "cv", number = 10)
rf_model <- train(is_recid ~ ., data = train_data, method = "rf",
                  trControl = train_control)

predictions <- predict(rf_model, newdata=test_data)
conf_matrix_rf_corr <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_rf_corr$byClass[2]
fp_rate <- conf_matrix_rf_corr$byClass[3]

precision <- conf_matrix_rf_corr$byClass[1]
recall <- conf_matrix_rf_corr$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_rf_corr <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_rf_corr)

#SVM
svmFit <- train(is_recid ~ ., data = train_data, method = "svmRadial",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10, 100)))

testPred <- predict(svmFit, newdata = test_data)
conf_matrix_SVM_corr <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_SVM_corr$byClass[2]
fp_rate <- conf_matrix_SVM_corr$byClass[3]

precision <- conf_matrix_SVM_corr$byClass[1]
recall <- conf_matrix_SVM_corr$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_SVM_corr <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                   f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_SVM_corr)

# Neural Net model

#setting up cross validation method
cv <- trainControl(method = "cv", number = 10)

#define the parameter grid to search over
grid <- expand.grid(size = 1:10, decay = c(0, 0.1, 1, 2))

nn_model <- train(is_recid ~ ., data = train_data, method = "nnet", tuneGrid = grid, trace=FALSE, maxit=100, MaxNWts = 1000, trControl = cv)

predictions <- predict(nn_model, test_data)

conf_matrix_NN_corr <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NN_corr$byClass[2]
fp_rate <- conf_matrix_NN_corr$byClass[3]

precision <- conf_matrix_NN_corr$byClass[1]
recall <- conf_matrix_NN_corr$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_nn_corr <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_nn_corr)

###################Forward Selection##################################
# Initialize empty set of attributes and vector to store selected attributes
train$is_recid <- as.numeric(train$is_recid)
selected <- c()
current <- NULL

# Loop through attributes and add to set
set.seed(123)
rsq_values <- list()

for (i in 1:(ncol(train)-1)) {
  candidate <- names(train)[i]
  if (!(candidate %in% selected)) {
    formula <- paste("is_recid ~", paste(c(selected, candidate), collapse = "+"))
    model <- lm(formula, data = train)
    if (is.null(current) || summary(model)$adj.r.squared > summary(current)$adj.r.squared) {
      current <- model
      selected <- c(selected, candidate)
      rsq_values[[candidate]] <- round(summary(model)$adj.r.squared, 3)
      cat("Added", candidate, "with adjusted R-squared", round(summary(model)$adj.r.squared, 3), "\n")
    }
  }
}

# Evaluate performance on test set
formula <- paste("is_recid ~", paste(selected, collapse = "+"))
model <- lm(formula, data = train)
pred <- predict(model, newdata = test)
mse <- mean((test$is_recid - pred)^2)
cat("Test MSE:", round(mse, 3))


library(dplyr)


# Sort data frame by adjusted R-squared value in descending order
rsq_values_sorted <- sort(sapply(rsq_values, function(x) x), decreasing = TRUE)
rsq_values_sorted

dataset1 <- dataset[c('end','days_b_screening_arrest','v_score_text','decile_score.1','is_recid')]
dataset1$is_recid <- as.factor(dataset1$is_recid)
#Weka
set.seed(123)
trainIndex <- createDataPartition(dataset1$is_recid, p = .8, list = FALSE)
train_data <- dataset1[ trainIndex,]
test_data <- dataset1[-trainIndex,]
ctrl <- trainControl(method="cv", number=10)
model <- train(is_recid ~ ., data=train_data, method="J48", trControl=ctrl)
predictions <- predict(model, newdata=test_data)
conf_matrix_Weka_FS <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_Weka_FS$byClass[2]
fp_rate <- conf_matrix_Weka_FS$byClass[3]

precision <- conf_matrix_Weka_FS$byClass[1]
recall <- conf_matrix_Weka_FS$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_weka_FS <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_weka_FS)

#Naive Bayes
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
nb_model <- train(is_recid ~ ., data = train_data, method = "nb",
                  trControl = train_control)

predictions <- predict(nb_model, newdata=test_data)
conf_matrix_NB_FS <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NB_FS$byClass[2]
fp_rate <- conf_matrix_NB_FS$byClass[3]

precision <- conf_matrix_NB_FS$byClass[1]
recall <- conf_matrix_NB_FS$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_NB_FS <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_NB_FS)

#Random Forest
train_control <- trainControl(method = "cv", number = 10)
rf_model <- train(is_recid ~ ., data = train_data, method = "rf",
                  trControl = train_control)

predictions <- predict(rf_model, newdata=test_data)
conf_matrix_rf_FS <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_rf_FS$byClass[2]
fp_rate <- conf_matrix_rf_FS$byClass[3]

precision <- conf_matrix_rf_FS$byClass[1]
recall <- conf_matrix_rf_FS$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_rf_FS <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_rf_FS)

#SVM
svmFit <- train(is_recid ~ ., data = train_data, method = "svmRadial",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10, 100)))

testPred <- predict(svmFit, newdata = test_data)
conf_matrix_SVM_FS <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_SVM_FS$byClass[2]
fp_rate <- conf_matrix_SVM_FS$byClass[3]

precision <- conf_matrix_SVM_FS$byClass[1]
recall <- conf_matrix_SVM_FS$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_SVM_FS <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                 f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_SVM_FS)

# Neural Net model

#setting up cross validation method
cv <- trainControl(method = "cv", number = 10)

#define the parameter grid to search over
grid <- expand.grid(size = 1:10, decay = c(0, 0.1, 1, 2))

nn_model <- train(is_recid ~ ., data = train_data, method = "nnet", tuneGrid = grid, trace=FALSE, maxit=100, MaxNWts = 1000, trControl = cv)

pred <- predict(nn_model, test_data)

conf_matrix_NN_FS <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NN_FS$byClass[2]
fp_rate <- conf_matrix_NN_FS$byClass[3]

precision <- conf_matrix_NN_FS$byClass[1]
recall <- conf_matrix_NN_FS$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_nn_FS <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_nn_FS)

###################Random Forest Feature Importance#####################
train$is_recid <- as.factor(train$is_recid)
model <- randomForest(is_recid ~ ., data=train, importance=TRUE)
importance <- importance(model)
head(importance)
significant.features <- sort(importance[,4], decreasing = TRUE)[1:4]
print(significant.features)
dataset1 <- dataset[c('end','age','priors_count', 'decile_score.1', 'is_recid')]
dataset1$is_recid <- as.factor(dataset1$is_recid)

#Weka
set.seed(123)
trainIndex <- createDataPartition(dataset1$is_recid, p = .8, list = FALSE)
train_data <- dataset1[ trainIndex,]
test_data <- dataset1[-trainIndex,]
ctrl <- trainControl(method="cv", number=10)
model <- train(is_recid ~ ., data=train_data, method="J48", trControl=ctrl)
predictions <- predict(model, newdata=test_data)
conf_matrix_Weka_RF <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_Weka_RF$byClass[2]
fp_rate <- conf_matrix_Weka_RF$byClass[3]

precision <- conf_matrix_Weka_RF$byClass[1]
recall <- conf_matrix_Weka_RF$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_weka_RF <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_weka_RF)

#Naive Bayes
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
nb_model <- train(is_recid ~ ., data = train_data, method = "nb",
                  trControl = train_control)

predictions <- predict(nb_model, newdata=test_data)
conf_matrix_NB_RF <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NB_RF$byClass[2]
fp_rate <- conf_matrix_NB_RF$byClass[3]

precision <- conf_matrix_NB_RF$byClass[1]
recall <- conf_matrix_NB_RF$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_NB_RF <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_NB_RF)

#Random Forest
train_control <- trainControl(method = "cv", number = 10)
rf_model <- train(is_recid ~ ., data = train_data, method = "rf",
                  trControl = train_control)

predictions <- predict(rf_model, newdata=test_data)
conf_matrix_rf_RF <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_rf_RF$byClass[2]
fp_rate <- conf_matrix_rf_RF$byClass[3]

precision <- conf_matrix_rf_RF$byClass[1]
recall <- conf_matrix_rf_RF$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_rf_RF <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_rf_RF)

#SVM
svmFit <- train(is_recid ~ ., data = train_data, method = "svmRadial",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10, 100)))

testPred <- predict(svmFit, newdata = test_data)
conf_matrix_SVM_RF <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_SVM_RF$byClass[2]
fp_rate <- conf_matrix_SVM_RF$byClass[3]

precision <- conf_matrix_SVM_RF$byClass[1]
recall <- conf_matrix_SVM_RF$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_SVM_RF <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                 f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_SVM_RF)

# Neural Net model

#setting up cross validation method
cv <- trainControl(method = "cv", number = 10)

#define the parameter grid to search over
grid <- expand.grid(size = 1:10, decay = c(0, 0.1, 1, 2))

nn_model <- train(is_recid ~ ., data = train_data, method = "nnet", tuneGrid = grid, trace=FALSE, maxit=100, MaxNWts = 1000, trControl = cv)

pred <- predict(nn_model, test_data)

conf_matrix_NN_RF <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NN_RF$byClass[2]
fp_rate <- conf_matrix_NN_RF$byClass[3]

precision <- conf_matrix_NN_RF$byClass[1]
recall <- conf_matrix_NN_RF$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_nn_RF <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_nn_RF)



##########################Backward Elimination########################
train$is_recid <- as.numeric(train$is_recid)

model <- lm(is_recid ~ ., data=train)
backward <- stepAIC(model, direction="backward")
significant.features <- names(coef(backward)[-1])
print(significant.features)
dataset1 <- dataset[,c('age','priors_count','days_b_screening_arrest','decile_score.1','v_score_text','end','is_recid')]

dataset1$is_recid <- as.factor(dataset1$is_recid)
#Weka
set.seed(123)
trainIndex <- createDataPartition(dataset1$is_recid, p = .8, list = FALSE)
train_data <- dataset1[ trainIndex,]
test_data <- dataset1[-trainIndex,]

write.csv(train_data, file = 'train_data_BE.csv')
write.csv(test_data, file = 'test_data_BE.csv')

ctrl <- trainControl(method="cv", number=10)
model <- train(is_recid ~ ., data=train_data, method="J48", trControl=ctrl)
predictions <- predict(model, newdata=test_data)
conf_matrix_Weka_BE <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_Weka_BE$byClass[2]
fp_rate <- conf_matrix_Weka_BE$byClass[3]

precision <- conf_matrix_Weka_BE$byClass[1]
recall <- conf_matrix_Weka_BE$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_weka_BE <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_weka_BE)

#Naive Bayes
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
nb_model <- train(is_recid ~ ., data = train_data, method = "nb",
                  trControl = train_control)

predictions <- predict(nb_model, newdata=test_data)
conf_matrix_NB_BE <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NB_BE$byClass[2]
fp_rate <- conf_matrix_NB_BE$byClass[3]

precision <- conf_matrix_NB_BE$byClass[1]
recall <- conf_matrix_NB_BE$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_NB_BE <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_NB_BE)

#Random Forest
train_control <- trainControl(method = "cv", number = 10)
rf_model <- train(is_recid ~ ., data = train_data, method = "rf",
                  trControl = train_control)

predictions <- predict(rf_model, newdata=test_data)
conf_matrix_rf_BE <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_rf_BE$byClass[2]
fp_rate <- conf_matrix_rf_BE$byClass[3]

precision <- conf_matrix_rf_BE$byClass[1]
recall <- conf_matrix_rf_BE$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_rf_BE <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_rf_BE)

#SVM
svmFit <- train(is_recid ~ ., data = train_data, method = "svmRadial",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10, 100)))

testPred <- predict(svmFit, newdata = test_data)
confusionMatrix(testPred, test_data$is_recid)
conf_matrix_SVM_BE <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_SVM_BE$byClass[2]
fp_rate <- conf_matrix_SVM_BE$byClass[3]

precision <- conf_matrix_SVM_BE$byClass[1]
recall <- conf_matrix_SVM_BE$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_SVM_BE <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                 f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_SVM_BE)

# Neural Net model

#setting up cross validation method
cv <- trainControl(method = "cv", number = 10)

#define the parameter grid to search over
grid <- expand.grid(size = 1:10, decay = c(0, 0.1, 1, 2))

nn_model <- train(is_recid ~ ., data = train_data, method = "nnet", tuneGrid = grid, trace=FALSE, maxit=100, MaxNWts = 1000, trControl = cv)

pred <- predict(nn_model, test_data)

conf_matrix_NN_BE <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NN_BE$byClass[2]
fp_rate <- conf_matrix_NN_BE$byClass[3]

precision <- conf_matrix_NN_BE$byClass[1]
recall <- conf_matrix_NN_BE$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_nn_BE <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_nn_BE)

##########################XGBoost#############################
library(xgboost)
xgb_model <- xgboost(data = as.matrix(train[, -ncol(train)]), label = train$is_recid, nrounds = 3)
xgb_imp <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_imp)
most_important_features <- sort(xgb_imp$Feature, decreasing = TRUE)[1:4]
most_important_features
dataset1 <- dataset[,c('v_score_text','v_decile_score','sex','score_text','is_recid')]
dataset1$is_recid <- as.factor(dataset1$is_recid)
#Weka
set.seed(123)
trainIndex <- createDataPartition(dataset1$is_recid, p = .8, list = FALSE)
train_data <- dataset1[ trainIndex,]
test_data <- dataset1[-trainIndex,]
ctrl <- trainControl(method="cv", number=10)
model <- train(is_recid ~ ., data=train_data, method="J48", trControl=ctrl)
predictions <- predict(model, newdata=test_data)
conf_matrix_Weka_xgb <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_Weka_xgb$byClass[2]
fp_rate <- conf_matrix_Weka_xgb$byClass[3]

precision <- conf_matrix_Weka_xgb$byClass[1]
recall <- conf_matrix_Weka_xgb$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_weka_xgb <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                   f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_weka_xgb)

#Naive Bayes
train_control <- trainControl(method = "cv", number = 10)

# Train the Naive Bayes model using 10-fold cross-validation
nb_model <- train(is_recid ~ ., data = train_data, method = "nb",
                  trControl = train_control)

predictions <- predict(nb_model, newdata=test_data)
conf_matrix_NB_xgb <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NB_xgb$byClass[2]
fp_rate <- conf_matrix_NB_xgb$byClass[3]

precision <- conf_matrix_NB_xgb$byClass[1]
recall <- conf_matrix_NB_xgb$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_NB_xgb <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                 f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_NB_xgb)

#Random Forest
train_control <- trainControl(method = "cv", number = 10)
rf_model <- train(is_recid ~ ., data = train_data, method = "rf",
                  trControl = train_control)

predictions <- predict(rf_model, newdata=test_data)
conf_matrix_rf_xgb <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_rf_xgb$byClass[2]
fp_rate <- conf_matrix_rf_xgb$byClass[3]

precision <- conf_matrix_rf_xgb$byClass[1]
recall <- conf_matrix_rf_xgb$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_rf_xgb <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                 f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_rf_xgb)

#SVM
svmFit <- train(is_recid ~ ., data = train_data, method = "svmRadial",
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = expand.grid(sigma = c(0.1, 1, 10), C = c(0.1, 1, 10, 100)))

testPred <- predict(svmFit, newdata = test_data)
conf_matrix_SVM_xgb <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_SVM_xgb$byClass[2]
fp_rate <- conf_matrix_SVM_xgb$byClass[3]

precision <- conf_matrix_SVM_xgb$byClass[1]
recall <- conf_matrix_SVM_xgb$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_SVM_xgb <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                  f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_SVM_xgb)

# Neural Net model

#setting up cross validation method
cv <- trainControl(method = "cv", number = 10)

#define the parameter grid to search over
grid <- expand.grid(size = 1:10, decay = c(0, 0.1, 1, 2))

nn_model <- train(is_recid ~ ., data = train_data, method = "nnet", tuneGrid = grid, trace=FALSE, maxit=100, MaxNWts = 1000, trControl = cv)

pred <- predict(nn_model, test_data)

conf_matrix_NN_xgb <- confusionMatrix(predictions, test_data$is_recid)

tp_rate <- conf_matrix_NN_xgb$byClass[2]
fp_rate <- conf_matrix_NN_xgb$byClass[3]

precision <- conf_matrix_NN_xgb$byClass[1]
recall <- conf_matrix_NN_xgb$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test_data$is_recid, as.numeric(predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(predictions), as.numeric(test_data$is_recid), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_nn_xgb <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
                 f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_nn_xgb)

result <- data.frame(perf_Weka, perf_NB, perf_rf, perf_SVM, perf_nn, perf_weka_corr, perf_NB_corr, perf_rf_corr, perf_SVM_corr, perf_nn_corr, perf_weka_FS, perf_NB_FS, perf_rf_FS, perf_SVM_FS, perf_nn_FS, perf_weka_RF, perf_NB_RF, perf_rf_RF, perf_SVM_RF, perf_nn_RF, perf_weka_BE, perf_NB_BE, perf_rf_BE, perf_SVM_BE, perf_nn_BE, perf_weka_xgb, perf_NB_xgb, perf_rf_xgb, perf_SVM_xgb, perf_nn_xgb)
t(result)
