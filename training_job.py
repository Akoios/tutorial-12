
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from gcloud import storage
import os

df = pd.read_csv("https://storage.googleapis.com/tutorial-datasets/hotel_reservations.csv")
df.head()

# Identify the categorical features in our data

categorical_features = ['Country','MarketSegment','ArrivalDateMonth','DepositType','CustomerType','IsCanceled']

df[categorical_features]=df[categorical_features].astype('category')

# Define the predictors (X) and the predicted variable (y)

y = df['IsCanceled']

X = df.drop(['IsCanceled'],axis=1)

# Encode the categorical variables

X_dum=pd.get_dummies(X,prefix_sep='-',drop_first=True)

# Split the dataset
X_dum = np.array(X_dum) 
y = np.array(y)

X_train,X_test,y_train,y_test= train_test_split(X_dum,y, test_size=.25,random_state=40)

# Prepare the GCloud storage

client = storage.Client()
bucket = client.get_bucket('tutorial-models')

# Train and store the Logistic Regression model

logistic=LogisticRegression()

logistic.fit(X_train,y_train)
filename='logistic_model.sav'
pickle.dump(logistic, open(filename, 'wb'))
blob = bucket.blob('logistic_model.sav')
blob.upload_from_filename('logistic_model.sav')
blob.make_public()

# Train and store the Random Forest model

rand=RandomForestClassifier(n_jobs=10, random_state=40)

rand.fit(X_train,y_train)
filename='random_forest_model.sav'
pickle.dump(logistic, open(filename, 'wb'))
blob = bucket.blob('random_forest_model.sav')
blob.upload_from_filename('random_forest_model.sav')
blob.make_public()

# Train and store the Gradient Boosting model

gb=GradientBoostingClassifier(random_state=50)

gb.fit(X_train,y_train)
filename='gradient_boosting_model.sav'
pickle.dump(logistic, open(filename, 'wb'))
blob = bucket.blob('gradient_boosting_model.sav')
blob.upload_from_filename('gradient_boosting_model.sav')
blob.make_public()

# Check the accuracy score for each model

y_pred= logistic.predict(X_test)
rand_pred=rand.predict(X_test)
gb_pred=gb.predict(X_test)

# Set and check accuracy threshold for deployment

accuracy_treshold = 0.70

if (accuracy_score(y_test,y_pred) < accuracy_treshold) or (accuracy_score(y_test,rand_pred) < accuracy_treshold) or (accuracy_score(y_test,gb_pred) < accuracy_treshold):
    print ("FALSE") > .ci_status/accuracy.txt


