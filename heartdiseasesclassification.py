import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os


data = pd.read_csv('D:\Personal Projects\Heart_Disease_project\Dataset\Raw dataset of heart disease.csv')
data.head()




imputer = SimpleImputer(strategy='most_frequent')
data['thal'] = imputer.fit_transform(data[['thal']]).ravel()

data.isnull().sum()



## converting Fasting blood sugar from Boolean data type to numerical
data['fasting blood sugar'] = data['fasting blood sugar'].map({False:0,True:1})

"""Converting Categorical columns to numerical"""

data['sex'] = data['sex'].map({'Male':0,'Female':1})
data['target'] = data['target'].map({'No':0,'Yes':1})
data['exercise induced angina'] = data['exercise induced angina'].map({'No':0,'Yes':1})

x = data.drop('target',axis=1)
y = data['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

y_train



num_cols = x_train.select_dtypes(exclude='object').columns
cat_cols = x_train.select_dtypes(include='object').columns

num_pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder())
    ])

preprocessor = ColumnTransformer([
    ('num_pipeline',num_pipeline,num_cols),
    ('cat_pipeline',cat_pipeline,cat_cols)
])

x_train_new = preprocessor.fit_transform(x_train)
x_test_new = preprocessor.transform(x_test)



models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'XGBoost': XGBClassifier(),
    'CatBoost': CatBoostClassifier(verbose=False)
}

params = {'Logistic Regression':{
    'penalty':['l2'],'C':[0.1,1,10]},
    'K-Nearest Neighbors':{
    'n_neighbors':[3,5,7,9],
    'weights':['uniform', 'distance'],'p':[1,2]},
    'Decision Tree':{
    'criterion':['gini', 'entropy'],
    'max_depth':[None, 5, 10,15,20]},'Random Forest':{
    'n_estimators':[100,200,300],'max_depth':[None,5,10]},
    'AdaBoost':{
    'n_estimators':[50,100,200],
    'learning_rate':[0.01,0.05,0.1]},
    'Gradient Boosting':{
    'n_estimators':[100,200,300],
    'learning_rate':[0.01,0.05,0.1]},
    'Support Vector Machine':{
    'C':[0.1,0.2,0.5],
    'kernel':['linear', 'rbf', 'poly']},
    'XGBoost':{
    'n_estimators':[100,200,300],
    'learning_rate':[0.01,0.05,0.1]},
    'CatBoost':{
    'iterations':[100,200,300],
    'learning_rate':[0.01,0.05,0.1]}}

for model_name,model in models.items():
  model = model
  model.fit(x_train_new,y_train)
  predicts = model.predict(x_test_new)
  print(f"{model_name} Accuracy: {accuracy_score(y_test,predicts)}")

print("=====================After Grid seach===================")
report = {}
for model_name,model in models.items():
  grid = GridSearchCV(model,param_grid=params[model_name],cv=5)
  grid.fit(x_train_new,y_train)
  report[model_name] = grid.best_score_
  print(f"{model_name} Best Score: {grid.best_score_}")

best_score = max(score for _,score in report.items())

best_model = [model for model,score in report.items() if score == best_score]
print(f"Best Model: {best_model[0]} with score {best_score}")



xgb_model = XGBClassifier(n_estimators=200,learning_rate=0.05)
xgb_model.fit(x_train_new,y_train)
predicts = xgb_model.predict(x_test_new) 

model_path = os.path.join('artifacts','xgboost.pkl')

preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

os.makedirs(os.path.dirname(model_path),exist_ok=True)

with open(model_path,'wb') as file:
  pickle.dump(xgb_model,file)

with open(preprocessor_path,'wb') as file:
  pickle.dump(preprocessor,file)


print(predicts)


"""# **Corrected column names to match the preprocessor's expectations**
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure',
       'serum cholestoral', 'fasting blood sugar',
       'resting electrocardiographic results ', 'maximum heart rate achieved',
       'exercise induced angina', 'oldpeak', ' slope of the peak',
       'colored by flourosopy', 'thal']

# Preprocess the input data using the same preprocessor fitted on the training data
input_data_df = pd.DataFrame(input_data, columns=columns)

# **Print input_data_df columns for verification**
print(input_data_df.columns)

input_data_processed = preprocessor.transform(input_data_df)

# Make a prediction using the trained XGBoost model
prediction = xgb_model.predict(input_data_processed)

print(f"Prediction: {prediction[0]}") # Output will be either 0 or 1"""
