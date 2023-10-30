from math import sqrt
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
        # rawData = rawData.drop(columns=['REF','Specific Bean Origin\nor Bar Name'])
        # rawData.columns = ['Company','ReviewDate','CocoaPercent','CompanyLocation','Rating','BeanType','BeanOrigin']
        # rawData['Company'] = [x.split('(')[0].strip() for x in rawData['Company']]
        # rawData['CocoaPercent'] = [float(x.split('%')[0].strip()) for x in rawData['CocoaPercent']]
        # rawData['CompanyLocation'] = [x.translate(str.maketrans('', '', string.punctuation)) for x in rawData['CompanyLocation']]
        # rawData['BeanType'] = rawData['BeanType'].replace('\xa0', np.nan)
        # rawData['BeanType'] = rawData['BeanType'].fillna('Unknown')
        # rawData['BeanType'] = [re.split(';|/|&|\.|,|\(',x)[0].strip() for x in rawData['BeanType']]
        # rawData['BeanOrigin'] = rawData['BeanOrigin'].replace('\xa0', np.nan)
        # rawData['BeanOrigin'] = rawData['BeanOrigin'].fillna('Unknown')
        # rawData['BeanOrigin'] = [re.split(';|/|&|\.|,|\(',x)[0].strip() for x in rawData['BeanOrigin']]
        # rawData.columns = ["MSSubClass", "MSZoning", "LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", 
        #                    "YearRemodAdd", "ExterQual", "ExterCond", "TotalBsmtSF", "GrLivArea", "TotRmsAbvGrd", "FullBath", 
        #                    "HalfBath", "GarageType", "GarageArea", "GarageQual", ]
        rawData = rawData.drop(columns=['Id','Street','Alley','Utilities','LandSlope','PoolQC','Fence','MiscFeature','FireplaceQu'])
        rawData.columns = ['MSSubClass','MSZoning','LotFrontage','LotArea','LotShape','LandContour','LotConfig',
                           'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                           'OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st',
                           'Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
                           'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                           'Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                           'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual',
                           'TotRmsAbvGrd','Functional','Fireplaces','GarageType','GarageYrBlt','GarageFinish',
                           'GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF',
                           'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal',
                           'MoSold','YrSold','SaleType','SaleCondition','SalePrice']
        rawData = rawData.dropna(inplace=False)
        rawData['LotFrontage'] = rawData['LotFrontage'].fillna(0)
        rawData['MasVnrType'] = rawData['MasVnrType'].fillna('None')
        rawData['MasVnrArea'] = rawData['MasVnrArea'].replace('NA', pd.NA).dropna()
        rawData['BsmtQual'] = rawData['BsmtQual'].fillna('NA')
        rawData['BsmtCond'] = rawData['BsmtCond'].fillna('NA')
        rawData['BsmtExposure'] = rawData['BsmtExposure'].fillna('NA')
        rawData['BsmtFinType1'] = rawData['BsmtFinType1'].fillna('NA')
        rawData['BsmtFinType2'] = rawData['BsmtFinType2'].fillna('NA')
        rawData['Electrical'] = rawData['Electrical'].fillna('NA')
        rawData['GarageType'] = rawData['GarageType'].fillna('NA')
        rawData['GarageYrBlt'] = pd.to_numeric(rawData['GarageYrBlt'].fillna(0))
        rawData['GarageFinish'] = rawData['GarageFinish'].fillna('NA')
        # categoricalColumns = ['Company','CompanyLocation','BeanType','BeanOrigin']
        categoricalColumns = ['MSZoning','LotShape','LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
                              'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                              'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',
                              'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                              'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                              'GarageType','GarageFinish','GarageQual','GarageCond', 'PavedDrive',
                              'SaleType','SaleCondition']
        
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

        column_head = pd.Index(['y']).append(XFit.columns)
        train=pd.DataFrame(np.column_stack([yFit,XFit]),columns=column_head)
        blind=pd.DataFrame(np.column_stack([yBlindTest,XBindTest]),columns=column_head)

        train.to_csv('api/lib/data/houseFitSample.csv', index=False)
        blind.to_csv('api/lib/data/houseBlindedTestSample.csv', index=False)

    @staticmethod
    def getBestPipeline(X,y):

        print('search_params 1')
        # search_params =  {'n_estimators': [128,256,512,1024], "learning_rate": [0.001,0.01,0.1], "max_depth" : [4,6,8]}
        search_params =  {'alpha':[0.01,0.05,0.1,0.5,1]}
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
        cv = GridSearchCV(pipe,search_params,cv=10,verbose=0,scoring='neg_mean_squared_error',n_jobs=-1,error_score=0.0)
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
