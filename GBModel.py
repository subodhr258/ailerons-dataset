#Code for recreating the model
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import joblib

df = pd.read_csv('ailerons_train.csv')
X = df.drop('goal',axis=1)
y = df['goal']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)

gbm = GradientBoostingRegressor(n_estimators=153,max_depth=5,min_samples_split=200,
min_samples_leaf=20,max_features=12,subsample=0.7)
gbm.fit(X_train,y_train)

y_pred = gbm.predict(X_test)
print("r2_score:",r2_score(y_test,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred)))

joblib.dump(gbm,'Gradient_Boosting_Model.h5')



