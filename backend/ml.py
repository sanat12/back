import pandas as pd
import yaml
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import numpy as np

#visualizatin
import seaborn as sn
import matplotlib.pyplot as plt

#model building
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class PredictDisease:    
    def __init__(self,symptoms):
        #loading the yaml config and initializing the model:
        #path=r"C:\Users\Sanat\Desktop\PredxHack2020\backend\config.yaml"
        """try: 
            with open(path,'r') as file:
                self.config = yaml.safe_load(file)
                #print("******************************************************************************")
        except Exception as e:
            print('Error reading the config file')"""
        self.verbose=False
        self.train_feat,self.train_labels, self.train_df,self.columns=self._loading_train()
        self.test_feat,self.test_labels, self.test_df=self._loading_test()
        self.model_save_path = r"C:\Users\Sanat\Desktop\PredxHack2020\backend"
        self.symptoms=symptoms
        #saving data visualization 
       # self._feature_correlation(data_frame=self.train_df, show_fig=False)
    
    #function to load train dataset:
    def _loading_train(self):
        df_train=pd.read_csv(r"C:\Users\Sanat\Desktop\PredxHack2020\backend\new_train.csv")
        #the last column is not needed and the 2nd last column gives us the y-labels 
        columns=df_train.columns[:-1]
        
        #the training feature vectors
        train_feat=df_train[columns]
        
        #the y labels for train set
        train_labels=df_train['prognosis']
            
        return train_feat, train_labels, df_train,columns
    
    #function to load the test data 
    def _loading_test(self):
        df_test=pd.read_csv(r"C:\Users\Sanat\Desktop\PredxHack2020\backend\new_test.csv")
        #the last column is not needed and the 2nd last column gives us the y-labels 
        columns=df_test.columns[:-1]
        
        #the test feature vectors
        test_feat=df_test[columns]
        
        #the y labels for test set
        test_labels=df_test['prognosis']
            
        return test_feat, test_labels, df_test
    
    #creating user defined test data:
    def create_test(self):
        symptoms=self.symptoms
        
        test=np.zeros(132,dtype=int)
        
        train=self.train_feat
        for s in symptoms:
            index_no=train.columns.get_loc(s)
            test[index_no]=1
        test=test.reshape(1,132)
        column_values=self.columns
        df = pd.DataFrame(data = test ,columns = column_values) 
        return df
        
    #model selection 
    def select_model(self):
        self.clf = MultinomialNB()
        return self.clf
    
    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_feat, self.train_labels,
                                                          test_size=0.2,
                                                          random_state=100,)

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
            
        return X_train, y_train, X_val, y_val
    
    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)
        
        # Save Trained Model
        filename=r"C:\Users\Sanat\Desktop\PredxHack2020\backend"+ "model_to_be_used"
        joblib.dump(classifier, filename + ".joblib")
        
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.corr()
        fig, ax = plt.subplots()
        fig.set_size_inches(14, 14)
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        #plt.show()
    
    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            filename=r"C:\Users\Sanat\Desktop\PredxHack2020\backend\saved_modelmnb"
            clf = joblib.load(filename +  ".joblib")
        except Exception as e:
            print("Model not found...")
        symptoms=self.symptoms
        if len(symptoms) is not 0:
            new_test=self.create_test()
            
            result = clf.predict(new_test)
            probs=clf.predict_proba(new_test)
            probs.reshape(1,-1)
            labels=clf.classes_
            preds=[]
            for i in probs[0]:
                predi=100*i
                preds.append(predi)
            x=dict(zip(labels,preds))
            top=dict(sorted(x.items(), key=lambda item: item[1],reverse=True))
            top = list(top.items())
            top=top[0:3]
            return top
            
        
        return []     

"""symps=list(map(str,mldata.split(",")))
if __name__=='__main__':
    #disease_model=PredictDisease(symps)
    disease_model.train_model()
    ans=disease_model.make_prediction()"""