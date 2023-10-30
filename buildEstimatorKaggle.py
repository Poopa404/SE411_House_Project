from math import sqrt
from matplotlib import pyplot as plt
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
            os.remove('api/lib/data/houseFitSample.csv')
            os.remove('api/lib/data/houseBlindedTestSample.csv')
        except OSError:
            pass

        # Cleaning the data
        rawData = pd.read_csv('api/lib/data/train.csv')
        # Find rows with NA
        df_na = rawData.isna().sum().to_frame().reset_index()
        df_na.columns = ["Column Name", "Null Count"]
        df_na["Null Percentage"] = round((df_na["Null Count"] / len(rawData))*100, 2)
        # Drop columns with NA more than 200
        df_many_na_columns = df_na[df_na["Null Count"] >= 200]["Column Name"].tolist()
        df_v1 = rawData.drop(columns=df_many_na_columns, axis=1)
        df_v2 = df_v1.dropna()
        # Id doesnt matter
        # Utilities only have one value
        rawData = df_v2.drop(columns=['Id', 'Utilities'])
        # Drop Outliers
        expensive_top2 = rawData.sort_values(by="SalePrice", ascending=False).head(2).index
        cheap_top2 = rawData.sort_values(by="SalePrice", ascending=True).head(2).index
        drop_index = expensive_top2.union(cheap_top2)
        df_v5 = rawData.drop(drop_index)
        print(df_v5.shape)
        # Numerical Variables
        numerical_columns = df_v5.select_dtypes(include=np.number).columns.tolist()
        # print(numerical_columns)
        # print(len(numerical_columns))
        corr_matrix = df_v5[numerical_columns].corr()
        # Low Correaltion ( < 0.2)
        low_corr = corr_matrix[corr_matrix["SalePrice"].abs() < 0.2].index.tolist()
        # print(low_corr)
        # print(len(low_corr))
        df_v6 = df_v5.drop(columns=low_corr, axis=1)
        print(df_v6.shape)
        # high Correlation ( >= 0.2)
        high_corr = corr_matrix[corr_matrix["SalePrice"].abs() >= 0.2].index.tolist()
        # print(high_corr)
        # print(len(high_corr))
        high_correlation = df_v6[high_corr].corr()
        df_v7 = df_v6.drop(columns=["TotalBsmtSF", "GarageYrBlt", "TotRmsAbvGrd"], axis=1)
        print(df_v7.shape)
        # Categorical Variables
        categoricalColumns = df_v7.select_dtypes(include=['object']).columns.tolist()
        print(categoricalColumns)
        print(len(categoricalColumns))
        # categoricalColumns = ['MSZoning','LotShape','LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
        #                       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        #                       'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',
        #                       'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        #                       'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
        #                       'GarageType','GarageFinish','GarageQual','GarageCond', 'PavedDrive',
        #                       'SaleType','SaleCondition']
        
        labelDict = {}
        for categoricalColumn in categoricalColumns:
            labelDict[categoricalColumn] = LabelEncoder()
            rawData[categoricalColumn] = labelDict[categoricalColumn].fit_transform(rawData[categoricalColumn])
            curLbl = labelDict[categoricalColumn].classes_.tolist()
            print(categoricalColumn)
            if 'Unknown' not in curLbl:
                 bisect.insort_left(curLbl, 'Unknown')
            
            labelDict[categoricalColumn].classes_ = curLbl

        leOutput =  os.path.join('api/lib/data/labelDict.pickle')
        file = open(leOutput,'wb')
        pickle.dump(labelDict,file)
        file.close()

        X = rawData.drop('SalePrice', axis=1)
        Y = rawData['SalePrice']
        XFit, XBindTest, yFit, yBlindTest =  train_test_split(X,Y,test_size = 0.3)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(XFit)
        X_test_scaled = scaler.fit_transform(XBindTest)

        column_head = pd.Index(['y']).append(XFit.columns)
        train=pd.DataFrame(np.column_stack([yFit,X_train_scaled]),columns=column_head)
        blind=pd.DataFrame(np.column_stack([yBlindTest,X_test_scaled]),columns=column_head)

        train.to_csv('api/lib/data/houseFitSample.csv', index=False)
        blind.to_csv('api/lib/data/houseBlindedTestSample.csv', index=False)

    @staticmethod
    def getBestPipeline(X,y):

        print('search_params 1')
        # search_params =  {'n_estimators': [128,256,512,1024], "learning_rate": [0.001,0.01,0.1], "max_depth" : [4,6,8]}
        search_params =  {'alpha':[0.001, 0.01, 0.1, 1, 10]}
        search_params = dict(('estimator__'+k, v) for k, v in search_params.items())
        print('search_params 2')
        search_params['normalizer'] = [None,StandardScaler()]
        search_params['featureSelector'] = [None,PCA(n_components=0.90, svd_solver='full')]
        print('pipe')
        pipe = Pipeline(steps=[
                ('normalizer',None),
                ('featureSelector', None),
                # ('estimator', xgb.XGBRegressor(objective='reg:squarederror'))
                ('estimator', Lasso(tol=1e-2))
        ])
        print('cv')
        cv = GridSearchCV(pipe,search_params,cv=5,verbose=0,scoring='neg_mean_squared_error',n_jobs=-1,error_score=0.0)
        cv.fit(X, y)

        return cv

    @staticmethod
    def createModel():
        print('fitting...')
        fit  = pd.read_csv('api/lib/data/houseFitSample.csv')
        XFit = fit.drop(['y'],axis=1)
        yFit = fit['y']

        print('testing...')
        blindTest  = pd.read_csv('api/lib/data/houseBlindedTestSample.csv')
        XBlindTest = blindTest.drop(['y'],axis=1)
        yBlindTest = blindTest['y']

        print('estimator')
        optimizedModel = BuildEstimator.getBestPipeline(XFit,yFit).best_estimator_
        yPredFit  = optimizedModel.predict(XFit)
        yPredTest = optimizedModel.predict(XBlindTest)

        fit_score = mean_squared_error(yFit,yPredFit)
        test_score = mean_squared_error(yBlindTest,yPredTest)

        print("fit mse = %.2f, test mse = %.2f" %(fit_score,test_score))
        print("fit rmse = %.2f, test rmse = %.2f" %(sqrt(fit_score),sqrt(test_score)))
        f = open('mseResult.txt','a')
        f.write("\nfit mse = %.2f, test mse = %.2f\nfit rmse = %.2f, test rmse = %.2f\n" %(fit_score,test_score,sqrt(fit_score),sqrt(test_score)))
        file = open('api/lib/model/house.pickle','wb')
        pickle.dump(optimizedModel,file)
        file.close()

if __name__ == '__main__':
    BuildEstimator.createBlindTestSamples()
    BuildEstimator.createModel()
    
    
