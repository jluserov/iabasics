import pandas
from sklearn import linear_model
import sklearn
from pylab import *
import numpy as np

df=pandas.read_csv("data.csv")
X=df[['Peso','Cantidad']]
y=df['CO2']


regr = linear_model.LinearRegression()
regr.fit(X, y)

predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)
