

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Hamza\Desktop\orthopedic patients\column_3C_weka.csv")

X=df.iloc[:,[0,1,2,3,4,5,6]].values   #all values

#pelvic_incidence x sacral_scope  0.81
x2=df.iloc[:,2].values.reshape(-1,1)
x3=df.iloc[:,3].values.reshape(-1,1)
y=df.iloc[:,0].values

#random forest regression
randforest=RandomForestRegressor(n_estimators=100,random_state=42)

###############  x2
randforest.fit(x2, y)
y_head2=randforest.predict(x2)
print("r2 score x2 : ",r2_score(y, y_head2))  #0.95

plt.plot(x2,y)
plt.scatter(x=x2, y=y_head2)

################  x3

randforest.fit(x3,y)
y_head3=randforest.predict(x3)
print("r2 score x3 : ",r2_score(y,y_head3))

plt.plot(x3,y)
plt.scatter(x=x3, y=y_head3)
