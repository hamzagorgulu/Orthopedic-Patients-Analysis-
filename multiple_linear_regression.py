
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv(r"C:\Users\Hamza\Desktop\orthopedic patients\column_3C_weka.csv")

X=df.iloc[:,[0,1,2,3,4,5,6]].values   #all values

#pelvic_incidence x sacral_scope  0.81
x=df.iloc[:,[2,3]].values
y=df.iloc[:,0].values.reshape(-1,1)


print(df.corr())

#sns.heatmap(df.corr(), annot=True, cmap='RdBu', center=0)   #corr otomatik yapabilirsin
#correlations
#pelvic_incidence x sacral_scope  0.81
#lumbar_lordosis_angle x pelvic_incidence   0.72

#multiple linear regression
multiple_reg=LinearRegression()
multiple_reg.fit(x, y)

print("b0: ",multiple_reg.intercept_)

print("b1 ,b2 : ",multiple_reg.coef_)

print("50 lordos 60 sacral slope: ",multiple_reg.predict(np.array([[50,60]]))," pelvic indicence")  
#50 lordoz 60 sacral slope: 73 pelvic incidence
y_head = multiple_reg.predict(x)

print("r2score: ",r2_score(y, y_head))


plt.plot(x,y)
plt.scatter(x=x, y=y_head)











