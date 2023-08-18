# Credit-Risk-Modelling
What is Credit Risk?
Credit risk is the possibility that a borrower will not be able to make timely payments and will default on their debt. It refers to the possibility that a lender may not get the interest or principal due to them on time.

Financial organizations are concerned about reducing the risk of default. As a result, commercial and investment banks, venture capital funds, asset management organizations, and insurance corporations, to mention a few, are increasingly depending on technology to anticipate which customers are most likely to default on their obligations.

What is Credit Risk Modelling?
A person’s credit risk is influenced by a variety of things. As a result, determining a borrower’s credit risk is a difficult undertaking. Credit risk modeling has entered the scene since there is so much money relying on our ability to appropriately predict a borrower’s credit risk.

Credit risk modeling is the practice of applying data models to determine two key factors. The first is the likelihood that the borrower will default on the loan. The second factor is the lender’s financial impact if the default occurs.

Credit risk models are used by financial organizations to assess the credit risk of potential borrowers.

For companies involved in the financial system, preserving the financial health of clients is critical. However, you could be wondering how to protect each client’s financial well-being. The answer to this challenge entails assessing each client’s payment likelihood based on a set of criteria and devising tactics to anticipate customer wants.

As a result, the goal of this research is to forecast the likelihood of default on a specific obligation, in this example, credit cards. This will enable the creation of solutions that reduce the risk of the client’s financial health deteriorating. Furthermore, it is proposed to employ clustering techniques to locate homogeneous portions within the population and thus provide differentiated treatment to each client in order to assist the creation of collection tactics.

The model will be constructed using the procedures below.

1. Data preparation and Pre-processing

2. Feature Engineering and Selection

3. Model Development and Model Evaluation


(Source: https://www.omnisci.com/technical-glossary/feature-engineering)
About the Data
The data set was taken from Kaggle here

This study analyses the predicted accuracy of probability of default among six data mining methods in the instance of consumers' default payments in Taiwan. The result of predictive accuracy of the projected probability of default will be more beneficial than the binary result of classification — credible or not credible clients — from the standpoint of risk management. Because the true likelihood of default is unknown, this study used a novel sorting Smoothing Method to approximate it. The response variable (Y) is the real probability of default, while the independent variable is the prediction chance of default (X).

Variables:

LIMIT_BAL: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. SEX: Gender (1 = male; 2 = female).

EDUCATION: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).

MARRIAGE: Marital status (1 = married; 2 = single; 3 = divorse, 0= others).

AGE: Age (year).

PAY0 — PAY6: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: PAY0 = the repayment status in September, 2005; PAY1 = the repayment status in August, 2005; . . .;PAY6 = the repayment status in April, 2005. The measurement scale for the repayment status is: -2: No consumption; -1: Paid in full; 0: The use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

BILL_AMT1- BILL_AMT6: Amount of bill statement (NT dollar). BILL_AMT1 = amount of bill statement in September, 2005; BILL_AMT2 = amount of bill statement in August, 2005; . . .; BILL_AMT6 = amount of bill statement in April, 2005.

PAY_AMT1-PAY_AMT2: Amount of previous payment (NT dollar). PAY_AMT1 = amount paid in September, 2005; PAY_AMT2 = amount paid in August, 2005; . . .; PAY_AMNT6 = amount paid in April, 2005.

# Reading the data 
credit_risk= pd.read_csv("UCI_credit_card.csv")
credit_risk.head()

Data
Data preparation and Pre-processing
This is where the data for credit risk modeling came from. Initially, the data revealed a total of 25 attributes. It is critical to clean the data in a suitable format before developing any machine learning model.

Analysis of column “EDUCATION”
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)

From the Data Description given, we know that in df.EDUCATION, 5 and 6 represents “unknown”
Changing 0,5 and 6 to keep it under 1 category.

df['EDUCATION'].replace({0:1,1:1,2:2,3:3,4:4,5:1,6:1}, inplace=True)
df.EDUCATION.value_counts()

Education column in data
Analysis of column “MARRIAGE”
Marital status (1=married, 2=single, 3=others)

# lets see the values count in column marriage
df['MARRIAGE'].value_counts()

Marriage column in data
Here I am going to map 0 with 1.

df['MARRIAGE'].replace({0:1,1:1,2:2,3:3}, inplace=True)
df['MARRIAGE'].value_counts()

marriage column after mapping
Analysis of column “PAY_0 to PAY_6”
PAY_0: Repayment status in September 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)


pay_0 column in data
Data Visualization
So, By looking at the target varibale we could say that, our data is pretty much imbalance. We would like to make it balanced before going to trained the model.
Target variable (Author image)

Education column (Author image)

Age column distribution (Author image)

Sex column ( Author image)

Scatter plot ( Author image)
Creating Independent features and dependent features
Independent variables (also referred to as Features) are the input for a process that is being analyzed. Dependent variables are the output of the process.
# Independnet features
X = df.drop(['default.payment.next.month'], axis=1)
# Dependent feature
y = df['default.payment.next.month']
X.head()
Scaling the features
The process of scaling or transforming all of the variables in our dataset to a given scale is known as feature scaling. Gradient descent optimization is used in some machine learning methods such as linear regression, logistic regression, and others. For these algorithms to work properly, the data must be scaled.
Standardization is the process of scaling the data values in such a way that they gain the properties of standard normal distribution. This means that the data is rescaled in such a way that the mean becomes zero and the data has a unit standard deviation.

Formula for standardization
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X= scaler.fit_transform(X)
Train Test Split
The entire dataset (population) is split into two groups: the train set and the test set. Depending on the use case, the data can be divided into 70–30 or 60–40, 75–25 or 80–20, or even 50–50. In general, the proportion of training data must be greater than the proportion of test data.

Splitting the data (Image Source: DataVedas)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=42)
Class imbalance
For over-sampling techniques, SMOTE (Synthetic Minority Oversampling Technique) is considered one of the most popular and influential data sampling algorithms in ML and data mining. With SMOTE, the minority class is over-sampled by creating “synthetic” examples rather than by over-sampling with replacement [2]. These introduced synthetic examples are based along the line segments joining a defined number of k minority class nearest neighbors, which is in the learning package is set at five by default.
from imblearn.over_sampling import SMOTE
from collections import Counter

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# define oversampling strategy
SMOTE= SMOTE()

# fit and apply the transform 
X_train,y_train= SMOTE.fit_resample(X_train,y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train))

Building Model
Logistic Regression
Random Forest Classifier
XGBoost Classifier
Cross-validation
Logistic regression Model
The main aim of logistic regression is to determine the relationship between features and the probability of a particular outcome.
It used the Sigmoid function to map the data.

Image source here
from sklearn.linear_model import LogisticRegression
logit= LogisticRegression()
logit.fit(X_train, y_train)
# Predicting the model
pred_logit= logit.predict(X_test)
Evaluating the logit model
Accuracy: the proportion of the total number of correct predictions.
Positive Predictive Value or Precision: the proportion of positive cases that were correctly identified.
Negative Predictive Value: the proportion of negative cases that were correctly identified.
Sensitivity or Recall: the proportion of actual positive cases which are correctly identified.
Specificity: the proportion of actual negative cases which are correctly identified.

Confusion metrics (Image source here)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, plot_confusion_matrix, plot_precision_recall_curve

print("The accuracy of logit model is:", accuracy_score(y_test, pred_logit))
print(classification_report(y_test, pred_logit))

Classification report of Logit model
Random Forest Classifer
Step 1: In Random forest n number of random records are taken from the data set having k number of records.

Step 2: Individual decision trees are constructed for each sample.

Step 3: Each decision tree will generate an output.

Step 4: Final output is considered based on Majority Voting or Averaging for Classification and regression respectively.


Random forest(Image source here)
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
# Fitting the model
rf.fit(X_train,y_train)
# Predicting the model
pred_rf= rf.predict(X_test)
Evaluating the Random Forest model
print("The accuracy of logit model is:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test,pred_rf ))

classification report of the Random Forest model
XGBoost Classifier
Another prominent boosting method is Extreme Gradient Boosting or XGBoost. XGBoost is, in reality, just a tweaked version of the GBM algorithm! The operation of XGBoost is identical to that of GBM. In XGBoost, trees are produced in sequential order, with each tree trying to fix the errors of the previous trees.

Image source here
import xgboost as xgb

xgb_clf= xgb.XGBClassifier()
#fitting the model
xgb_clf.fit(X_train,y_train)
## Predicting the model
xgb_predict= xgb_clf.predict(X_test)
Evaluating the Xgboost model

classification report of xgboost model
Hyperparameter tunning the xgboost model
## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
random_search=RandomizedSearchCV(xgb_clf,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

# fitting the RandomizedSearchCV
random_search.fit(X_train,y_train)
Then we will find the best estimators and parameters.

# Finding the best estimators
random_search.best_estimator_
# Finding the best param
random_search.best_params_

Optimal parameters

Best estimators
# Predicting model
y_pred= classifier.predict(X_test)
k-fold Cross-validation
Let’s extrapolate the last example to k-fold from 2-fold cross-validation. Now, we will try to visualize how does a k-fold validation work.


(Image source here)
This is a seven-fold cross-validation procedure.

Here’s how it works: we divide the total population into 5 equal samples. Models are now trained on four samples (blue boxes) and validated on one sample (yellow box). The model is then trained with a new sample held as validation in the second iteration. We developed a model on each sample in 5 iterations and held each of them as validation. This is a method for lowering selection bias and lowering prediction power variance. Once we have all five models, we average the error terms to determine which one is the best.

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)
score.mean()

Final accuracy
