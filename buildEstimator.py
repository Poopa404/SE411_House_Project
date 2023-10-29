from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from  sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import string
import bisect
import warnings
import re

class BuildEstimator:

    @staticmethod
    def createBlindTestSamples():

        try:
            os.remove('api/lib/data/fitSample.csv')
            os.remove('api/lib/data/blindedTestSample.csv')
        except OSError:
            pass

        # Cleaning the data
        rawData = pd.read_csv('api/lib/data/flavors_of_cacao.csv')
        rawData = rawData.drop(columns=['REF','Specific Bean Origin\nor Bar Name'])
        rawData.columns = ['Company','ReviewDate','CocoaPercent','CompanyLocation','Rating','BeanType','BeanOrigin']
        rawData['Company'] = [x.split('(')[0].strip() for x in rawData['Company']]
        rawData['CocoaPercent'] = [float(x.split('%')[0].strip()) for x in rawData['CocoaPercent']]
        rawData['CompanyLocation'] = [x.translate(str.maketrans('', '', string.punctuation)) for x in rawData['CompanyLocation']]
        rawData['BeanType'] = rawData['BeanType'].replace('\xa0', np.nan)
        rawData['BeanType'] = rawData['BeanType'].fillna('Unknown')
        rawData['BeanType'] = [re.split(';|/|&|\.|,|\(',x)[0].strip() for x in rawData['BeanType']]
        rawData['BeanOrigin'] = rawData['BeanOrigin'].replace('\xa0', np.nan)
        rawData['BeanOrigin'] = rawData['BeanOrigin'].fillna('Unknown')
        rawData['BeanOrigin'] = [re.split(';|/|&|\.|,|\(',x)[0].strip() for x in rawData['BeanOrigin']]

        categoricalColumns = ['Company','CompanyLocation','BeanType','BeanOrigin']

        labelDict = {}
        for categoricalColumn in categoricalColumns:
            labelDict[categoricalColumn] = LabelEncoder()
            rawData[categoricalColumn] = labelDict[categoricalColumn].fit_transform(rawData[categoricalColumn])
            curLbl = labelDict[categoricalColumn].classes_.tolist()
            if 'Unknown' not in curLbl:
                 bisect.insort_left(curLbl, 'Unknown')
            labelDict[categoricalColumn].classes_ = curLbl

        leOutput =  os.path.join('api/lib/data/labelDict.pickle')
        file = open(leOutput,'wb')
        pickle.dump(labelDict,file)
        file.close()

        X = rawData.drop('Rating', axis=1)
        Y = rawData['Rating']
        XFit, XBindTest, yFit, yBlindTest =  train_test_split(X,Y,test_size = 0.3)

        column_head = pd.Index(['y']).append(XFit.columns)
        train=pd.DataFrame(np.column_stack([yFit,XFit]),columns=column_head)
        blind=pd.DataFrame(np.column_stack([yBlindTest,XBindTest]),columns=column_head)

        train.to_csv('api/lib/data/fitSample.csv', index=False)
        blind.to_csv('api/lib/data/blindedTestSample.csv', index=False)

    @staticmethod
    def getBestPipeline(X,y):

        # search_params =  {'n_estimators': [128,256,512,1024], "learning_rate": [0.001,0.01,0.1], "max_depth" : [4,6,8]}
        search_params =  {'alpha':[0.01,0.05,0.1,0.5,1]}
        search_params = dict(('estimator__'+k, v) for k, v in search_params.items())

        search_params['normalizer'] = [None,StandardScaler()]
        search_params['featureSelector'] = [None,PCA(n_components=0.90, svd_solver='full')]

        pipe = Pipeline(steps=[
                ('normalizer',None),
                ('featureSelector', None),
                # ('estimator', xgb.XGBRegressor(objective='reg:squarederror'))
                ('estimator', Lasso())
        ])

        cv = GridSearchCV(pipe,search_params,cv=10,verbose=0,scoring='neg_mean_squared_error',n_jobs=-1,error_score=0.0)
        cv.fit(X, y)

        return cv

    @staticmethod
    def createModel():

        fit  = pd.read_csv('api/lib/data/fitSample.csv')
        XFit = fit.drop(['y'],axis=1)
        yFit = fit['y']

        blindTest  = pd.read_csv('api/lib/data/blindedTestSample.csv')
        XBlindTest = blindTest.drop(['y'],axis=1)
        yBlindTest = blindTest['y']

        optimizedModel = BuildEstimator.getBestPipeline(XFit,yFit).best_estimator_
        yPredFit  = optimizedModel.predict(XFit)
        yPredTest = optimizedModel.predict(XBlindTest)

        fit_score = mean_squared_error(yFit,yPredFit)
        test_score = mean_squared_error(yBlindTest,yPredTest)

        print("fit mse = %.2f and test mse = %.2f" %(fit_score,test_score))

        file = open('api/lib/model/flavors_of_cacao.pickle','wb')
        pickle.dump(optimizedModel,file)
        file.close()

if __name__ == '__main__':

    BuildEstimator.createBlindTestSamples()
    BuildEstimator.createModel()
