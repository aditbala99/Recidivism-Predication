# README

## 1. DATASET
ProPublica obtained a dataset of pretrial defendants and probationers from Broward County, FL, who had been assessed with the COMPAS screening system between January 1, 2013, and December 31, 2014. COMPAS recidivism risk scores are based on a defendant’s answers to the COMPAS screening survey. The survey is completed by pre-trial services in cooperation with the defendant after his or her arrest. The COMPAS survey, at least in the ProPublica data, is typically administered the same day or the day after a person is jailed. For the more than 11 thousand pretrial defendants in this dataset, ProPublica then collected data on future arrests through the end of March 2016, in order to study how the COMPAS score predicts recidivism for these defendants. ProPublica collected the data for its study and created a (Python) database. From that database, it constructed various sub-datasets that merged and calculated various important features.

The dataset used has 18 different important attributes and more than 16,000 tuples.
TARGET VARIABLE –> “is_recid”
An insight into the dataset is given below:

## 2. DATA CLEANING
For any sort of data analysis, a clean dataset is necessary. The four steps below outline the process:
Data cleaning is a critical step in the data preparation process, where data is reviewed and analyzed for errors, inconsistencies, and inaccuracies. Data cleaning helps to ensure that data is accurate, complete, and reliable. Here are some common methods for data cleaning:

1. Handling Missing Values:
Missing data can be a significant problem in data cleaning. Missing values can be handled by either imputing missing values or deleting the observations with missing data.

2. Standardizing Data:
Standardizing data involves ensuring that data values are in a consistent format. For example, ensuring that all dates are in the same format.

3. Outlier Detection:
Outliers are observations that are significantly different from other observations in the dataset. Identifying and removing outliers can help to improve the accuracy of the data.

## 3. ATTRIBUTE SELECTION METHODS
Attribute selection methods are used in machine learning and data mining to select a subset of relevant features (also called attributes or variables) from a larger set of available features. The objective of attribute selection is to improve the accuracy, efficiency, and interpretability of a machine learning model by reducing the dimensionality of the input space. We have used the following attribute selection methods for the project:

1. Correlation-based:
Correlation-based attribute selection is a technique used in machine learning and data mining to select a subset of relevant features from a larger set of features or attributes. The technique involves calculating the correlation between each feature and the target variable (i.e., the variable to be predicted), and selecting the features that have the highest correlation with the target variable. In this technique, features with low correlation are discarded, as they are deemed to have little impact on the target variable. This helps to reduce the dimensionality of the data and improve the accuracy and efficiency of the predictive model. Correlation-based attribute selection can be applied to both numerical and categorical data, and there are several metrics used to calculate correlation, such as Pearson correlation coefficient, Spearman's rank correlation, and Kendall's tau correlation. The choice of metric depends on the type of data and the research question being addressed.

2. Forward selection:
Forward selection is a feature selection technique in machine learning and data mining. It involves starting with an empty set of features and iteratively adding one feature at a time to the model based on its performance until a desired level of performance or a maximum number of features is reached. The basic idea is to start with the most important feature and then add other features in a stepwise manner, evaluating the performance of the model at each step. At each iteration, the feature that leads to the greatest improvement in performance is selected and added to the feature set. Forward selection is a simple and efficient technique that can be used to identify the most relevant features for a particular problem. However, it may not always find the optimal subset of features, as it may get stuck in a local optimum or include irrelevant features early in the process that cannot be removed later.

3. Random Forest:
Random forest attribute selection is a method used to identify the most important variables in a dataset. It is based on the random forest algorithm, which builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. During the process, each decision tree is trained on a randomly selected subset of the features, and the importance of each feature is calculated based on how much it contributes to the overall prediction accuracy. The feature importance scores are then used to rank the variables in order of significance, and the top-ranked variables are selected for further analysis or modeling. Random forest attribute selection is a useful technique for reducing the dimensionality of a dataset and improving model performance by focusing on the most informative features.

4. Backward Elimination:
Backward elimination attribute selection is a feature selection technique used in machine learning and data mining to reduce the number of input variables or features in a dataset. This technique starts with all the features and iteratively removes the least significant feature until a stopping criterion is reached. The stopping criterion could be based on a predefined significance level, performance measure, or cross-validation score. The backward elimination process involves training a model with all the features, evaluating the significance of each feature using statistical tests, and removing the least significant feature from the model. The process continues until the stopping criterion is met, and the remaining subset of features is considered the optimal set. This technique is computationally expensive, especially for large datasets with many features, but it can significantly improve the model's performance by reducing overfitting and increasing generalization.

5. Extreme Gradient Boosting (XGBoost):
XGBoost (Extreme Gradient Boosting) is a popular machine learning algorithm used for regression and classification tasks. One of the key features of XGBoost is its ability to perform feature selection, which allows the algorithm to select the most important features from a given dataset. There are two main methods for feature selection in XGBoost:

   - Split-based feature importance: This method calculates the importance of each feature based on how much each feature reduces the loss function of the model when it is used in a split. Features that result in a large reduction in the loss function are considered more important.

   - Permutation feature importance: This method calculates the importance of each feature by permuting its values and measuring how much this affects the accuracy of the model. Features that result in a large drop in accuracy when their values are permuted are considered more important. Both methods can be used to select the most important features for a given dataset, which can help improve the accuracy and speed of the XGBoost model.

## 4. PROCEDURE
1. In the Analysis section of the R code, the first step involved defining the classification algorithms that we would use, such as Naive-Bayes, Weka, Neural Net, Support Machine Vector, and Random Forest. The next step involved implementing 10-fold cross-validation on the entire dataset using these five classification algorithms. We then collected the average metrics for accuracy, true positive rate, false positive rate, precision, recall, F1-measure, Matthews correlation coefficient (MCC), and ROC AUC for the 10 folds to evaluate the effectiveness of the classification algorithms.

2. The pre-processing steps involved removing null values and outliers, which improved the proficiency of the machine learning models trained on the data. This suggests that these steps were effective in improving the quality of the dataset.

3. Five machine learning models were selected - Weka, Random Forest, Support Vector Machine (SVM), Neural Nets, and Naïve Bayes - to predict the likelihood of an individual returning to prison after release. This implies that the problem being tackled was a classification problem.

4. The dataset was split into training and testing data using a 66-34 split, and the models were trained on the training data and evaluated on the test data using a confusion matrix. This suggests that the models were properly validated using a holdout method.

5. Among the five models, Naïve Bayes had the highest accuracy of 84.61% on the complete dataset, indicating that it performed the best in predicting the likelihood of an individual returning to prison after release.

6. Five attribute selection techniques were used - Forward Selection, Backward Elimination, Correlation Based Feature Selection, Random Forest Feature Selection, and XGBoost - to obtain subsets of the most important attributes from the parent dataset. These attribute selection techniques were used to identify the most relevant features to include in the models, which can help improve the performance of the models.

7. For each of the subsets obtained from the attribute selection techniques, the data was split into training and testing sets, and the five machine learning models were trained on the new training dataset. In total, 25 models were trained and evaluated.

8. Based on the results obtained from the 25 models, it was found that the Backward Elimination technique had the highest accuracy. This implies that this technique was effective in identifying the most important attributes for predicting the likelihood of an individual returning to prison after release.

## 5. RESULTS

### Performance of the model on the entire dataset with 10-fold Cross Validation
| Model         | Accuracy |
|---------------|----------|
| Weka          | 67.03%   |
| Naïve Bayes   | 84.61%   |
| Random Forest | 67.03%   |
| SVM           | 67.03%   |
| Neural Nets   | 71.42%   |

From this, we can say that out of the 5 models, Naïve Bayes gave the best results after 10 cross validations.

### Performance of the models after Attribute Selection
| Model         | Correlation-based | Forward Selection | Random Forest | Backward Elimination | XGBoost |
|---------------|-------------------|-------------------|---------------|-----------------------|---------|
| Weka          | 59.34%            | 69.23%            | 59.34%        | 79.12%                | 30.76%  |
| Naïve Bayes   | 75.82%            | 82.41%            | 75.8%         | 84.61%                | 67.03%  |
| Random Forest | 69.23%            | 67.03%            | 67.03%        | 69.23%                | 70.03%  |
| SVM           | 69.23%            | 67.03%            | 67.03%        | 69.23%                | 70.03%  |
| Neural Nets   | 73.62%            | 67.03%            | 67.03%        | 69.23%                | 70.03%  |

As seen from the model, we can say that for our dataset the Backward Elimination Attribute Selection worked the best as compared to the other 4.

## 6. CONCLUSION
The conclusion drawn from the project highlights several important aspects of machine learning. Firstly, the significance of feature selection in machine learning is emphasized. Feature selection techniques enable the identification of relevant features that contribute most to the prediction task, which can reduce the time taken to train machine learning models, while improving their performance. This is crucial as the quality and quantity of data used for model training heavily influence the accuracy of the classifiers.

The project also demonstrates how proficiency in training machine learning models can be improved using the R programming language. R is a widely used programming language for data analysis and statistical computing, and its popularity is due to its ease of use and powerful analytical capabilities. The project's experience in using R for machine learning can be useful for future projects, as it enables efficient data manipulation, feature selection, and model development, which can significantly enhance the performance of machine learning models.

The project also provided hands-on experience in data pre-processing, model selection, and evaluation. Data pre-processing is a critical step in machine learning, as it ensures that the data is consistent, accurate, and complete, thereby improving the performance of machine learning models. Model selection and evaluation are also crucial, as it involves selecting the appropriate algorithm for the task at hand and assessing its performance. Through the project, the team has gained practical experience in these areas, which can be invaluable in future machine learning projects.

Furthermore, the project has provided a deeper understanding of the strengths and limitations of different classification models and how to optimize their performance. By comparing and contrasting the performance of five different machine learning algorithms, including Weka, Random Forest, Support Vector Machine (SVM), Neural Nets, and Naïve Bayes, the team gained insights into the trade-offs between different algorithms and how to choose the most suitable algorithm for a specific task.

In conclusion, the project has provided valuable practical experience in various aspects of machine learning, including data pre-processing, feature selection, model selection, and evaluation, using the R programming language. The team gained a deeper understanding of the strengths and limitations of different classification models and how to optimize their performance, which can be useful for future machine learning projects in diverse domains.

Overall, the results suggest that the use of attribute selection techniques can help improve the performance of machine learning models in predicting the likelihood of an individual returning to prison after release, and that the Backward Elimination technique was the most effective in this case. However, it is important to note that the results obtained may be specific to this dataset and may not necessarily generalize to other datasets. Further research is needed to validate the effectiveness of these techniques on other datasets.

**BEST MODEL: NAÏVE BAYES USING BACKWARD ELIMINATION ATTRIBUTE SELECTION**

**Best Feature Selection Method:**
Based on the tables, it seems that the Backward Elimination feature selection method generally performs the best across the different models and evaluation metrics. The Backward Elimination model has the highest precision and recall scores for most models. It also has the highest accuracy scores for 4/5 models in the accuracy table. We can also observe that Forward Selection Feature Selection is quite consistent across all models.

**Best Model:**
Looking at the average accuracy, precision, and recall values for each model, we can see that the Naïve Bayes model consistently performs well across all attribute selection methods, followed by the Neural Nets and Support Vector Machine and Random Forest Classification models. Weka models generally have lower average performance across all attribute selection methods. Therefore, we can conclude that the Naïve Bayes model is the best performing model across all classifiers.

