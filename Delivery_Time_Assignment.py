# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:56:10 2021

@author: amart
"""

from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf

dat=pd.read_csv("F:\Data Science Assignments\Python-Assignment\Linear Regression\delivery_time.csv")
new=dat
new=new.rename(columns={'Delivery Time':'delt','Sorting Time':'sort'})
new.corr()
plt.hist(new.delt)
plt.boxplot(new.delt)
plt.plot(new.sort,new.delt,"ro");plt.xlabel("Sort Time");plt.ylabel("Delivery Time")
plt.hist(new.sort)


model=smf.ols("delt~sort",data=new).fit()
model.summary()             #Rsquare = 0.68 and Adj Rsq=0.666
pred1=model.predict(new)
pred1.corr(new.delt)        #0.8259972607955325
plt.scatter(x=new.sort,y=new.delt,color='green');plt.plot(new.sort,pred1,color='blue');plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")
rmse1=sqrt(mean_squared_error(new.delt, pred1))
rmse1                       #2.7916503270617654 
model.conf_int(0.05)

 
model2=smf.ols("delt~ np.log(sort)",data=new).fit()
model2.summary()             #Rsquare = 0.695 and Adj Rsq=0.679
pred2=model2.predict(new)
pred2.corr(new.delt)         #0.8339325279256244
plt.scatter(x=new.sort,y=new.delt,color='green');plt.plot(new.sort,pred2,color='blue');plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")
rmse2=sqrt(mean_squared_error(new.delt, pred2))
rmse2                        #2.733
model2.conf_int(0.05)


model3=smf.ols("delt~(sort+sort^2)",data=new).fit()
model3.summary()             #Rsquare =0.698   and   Adj Rsq =0.665
pred3=model3.predict(new)
pred3.corr(new.delt)         #0.835694279409746
plt.scatter(x=new['sort'],y=new['delt'],color='green');plt.plot(new['sort'],pred3,color='red');plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")
rmse3=sqrt(mean_squared_error(new.delt, pred3))
rmse3                        #2.7199406964726003

# *Model3 seems to be best as rmse value is less and the Rsquare value is more.