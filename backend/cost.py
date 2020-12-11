

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[38]:


class HealthExpenditure():
    def __init__(self,features):
        self.train=pd.read_csv(r"C:\Users\Sanat\Desktop\PredxHack2020\backend\new_train1.csv")
        
        self.features=features
        
        self.x_train=self.train.drop(["HEALTHEXP"],axis=1)
        self.y_train=self.train["HEALTHEXP"]
        
        log = np.log(self.y_train)
        log[log<0] = 0
        log = log/np.log(3)
        self.y_train = log
        
        self.categorical_features = self.get_categorical_features()
        self.numerical_features= [f for f in self.x_train.columns if f not in self.categorical_features]
        
        self.preprocessor=self.get_preprocessor()
        self.regressor=self.get_regressor()
     
    def create_test(self):
        features=self.features
        features=np.array(features)
        #print(features)
        features=features.reshape(1,-1)
        
        cols=['AGE31X', 'RACE3', 'GENDER', 'MARRY31X', 'RTHLTH31', 'MNHLTH31',
           'HIBPDX', 'MIDX', 'STRKDX', 'CHOLDX', 'CANCERDX', 'DIABDX', 'ARTHDX',
           'ASTHDX', 'ADHDADDX', 'PREGNT31', 'WLKLIM31', 'SOCLIM31', 'ADSMOK42',
           'K6SUM42', 'DFSEE42', 'EMPST31', 'INCOME_M', 'INSCOV15', 'POVCAT15']
        
        test_data = pd.DataFrame(features, columns =cols)
        return test_data
    
    def get_categorical_features(self):
        num_unique = self.x_train.nunique()
        categorical_features = num_unique[num_unique <= 10].index.tolist()
        for col in ["POVCAT15", "RTHLTH31", "MNHLTH31"]:
            categorical_features.remove(col)
        return categorical_features
    
    def get_preprocessor(self):
        categorical_transformer = Pipeline(steps = [("onehot", OneHotEncoder(handle_unknown = "ignore"))])
        numerical_transformer = Pipeline(steps = [ ("scaler", StandardScaler())])
        
        preprocessor = ColumnTransformer(transformers = [("cat", categorical_transformer,self.categorical_features),
                                                 ("num", numerical_transformer,self.numerical_features)])
        
        return preprocessor
    
    def get_regressor(self):
        regressor = GradientBoostingRegressor(n_estimators = 76, 
                                    max_depth = 5,
                                    min_samples_split = 2,
                                    min_samples_leaf = 5,
                                    random_state = 123)
        return regressor
                    
    def train_model(self):
        reg_xgb = Pipeline(steps = [("preprocessor", self.preprocessor),("regressor", self.regressor)])
        reg_xgb.fit(self.x_train,self.y_train)
        
        #print("model is trained")
        pickle.dump(reg_xgb, open(r"C:\Users\Sanat\Desktop\PredxHack2020\backend\cost.pickle", "wb"))
    
    def predict(self):
        pickle_in = open(r"C:\Users\Sanat\Desktop\PredxHack2020\backend\cost.pickle", "rb")
        reg_xgb = pickle.load(pickle_in)
        
        X_test=self.create_test()
        y_pred = reg_xgb.predict(X_test)
        
        ans=np.log(3)*y_pred
        ans=np.exp(ans)
        return (ans[0])


# In[32]:


# In[ ]:




