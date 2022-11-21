import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
#%matplotlib inline

smoke=pd.read_csv("/smoke.csv")

smoke.describe().transpose()
smoke.corr()
smoke.corr()['Fire Alarm'].sort_values()
smoke.corr()['Fire Alarm'].sort_values().plot(kind='bar')
cols=['Unnamed: 0' , 'CNT' , 'Raw Ethanol', 'Pressure[hPa]', 'UTC' ,
    'Humidity[%]' , 'PM1.0', 'PM1.0' , 'NC0.5' ,'NC1.0' ,'NC2.5']
smoke.drop(cols, axis=1, inplace=True) 

smoke["Fire Alarm"].value_counts()
smoke.info()

#Null vaule detection

smoke.isnull().sum()

#Outlier Detection

smoke.plot(kind='box', subplots=True, layout=(8,5), figsize=(17,20))
col = smoke.columns

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range

smoke_df=smoke.copy()
df0=smoke[smoke['Fire Alarm']==0.0]
df1=smoke[smoke['Fire Alarm']==1.0]

dec={"Colume":[],"outliers":[],"u_range":[] , 'l_range':[] ,"upper":[] , 'lower':[] ,
     "Nu_range":[] , 'Nl_range':[] ,"N_upper":[] , 'N_lower':[] 
    }
for column in smoke.iloc[:,:-1].columns:
    lr,ur=remove_outlier(df0[column])
    u_data=(smoke_df[(smoke_df['Fire Alarm']==0)&(smoke_df[column] > ur)])[column]
    l_data=(smoke_df[(smoke_df['Fire Alarm']==0)&(smoke_df[column] < lr)])[column]
    dec['Colume'].append(column + " 0")
    dec['outliers'].append(len(u_data)+len(l_data))
    dec['upper'].append(len(u_data))
    dec['lower'].append(len(l_data))
    dec['u_range'].append(ur)
    dec['l_range'].append(lr)
    
    if not u_data.empty:
        u_data=sorted(u_data)
        index=int(round(len(u_data)*0.7,0))
        ur=u_data[index]
        dec['N_upper'].append(len(u_data[index:]))
    else:
        dec['N_upper'].append(len(u_data))
        
    if not l_data.empty:
        l_data=sorted(l_data)
        index=int(round(len(l_data)*0.3,0))
        lr=l_data[index]
        dec['N_lower'].append(len(l_data[:index]))
    else:
        dec['N_lower'].append(len(l_data))
        
    index=(smoke_df[(smoke_df['Fire Alarm']==0)&((smoke_df[column] < lr)|(smoke_df[column] > ur))]).index
    index=index.to_list()
    
    dec['Nu_range'].append(ur)
    dec['Nl_range'].append(lr)
    
    smoke_df.drop(index,inplace=True)
    
    lr,ur=remove_outlier(df1[column])
    u_data=(smoke_df[(smoke_df['Fire Alarm']==1)&(smoke_df[column] > ur)])[column]
    l_data=(smoke_df[(smoke_df['Fire Alarm']==1)&(smoke_df[column] < lr)])[column]
    dec['Colume'].append(column + " 1")
    dec['outliers'].append(len(u_data)+len(l_data))
    dec['upper'].append(len(u_data))
    dec['lower'].append(len(l_data))
    dec['u_range'].append(ur)
    dec['l_range'].append(lr)
    
    if not u_data.empty:
        u_data=sorted(u_data)
        index=int(round(len(u_data)*0.7,0))
        ur=u_data[index]
        dec['N_upper'].append(len(u_data[index:]))
    else:
        dec['N_upper'].append(len(u_data))
        
    if not l_data.empty:
        l_data=sorted(l_data)
        index=int(round(len(l_data)*0.3,0))
        lr=l_data[index]
        dec['N_lower'].append(len(l_data[:index]))
    else:
        dec['N_lower'].append(len(l_data))
        
        
    index=(smoke_df[(smoke_df['Fire Alarm']==1)&((smoke_df[column] < lr)|(smoke_df[column] > ur))]).index
    index=index.to_list()
    
    dec['Nu_range'].append(ur)
    dec['Nl_range'].append(lr)
    
    smoke_df.drop(index,inplace=True)
    
x = smoke_df.drop("Fire Alarm", axis = 1).values
y = smoke_df['Fire Alarm'].values 
x2 = smoke.drop("Fire Alarm", axis = 1).values
y2 = smoke['Fire Alarm'].values 

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
X2_train,X2_test,Y2_train,Y2_test = train_test_split(x2,y2,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X2_train = sc.fit_transform(X_train)
X2_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

history=ann.fit(x=X_train,y=Y_train,validation_data=(X_test,Y_test),batch_size=20,epochs=15)
history2=ann.fit(x=X2_train,y=Y2_train,validation_data=(X2_test,Y2_test),batch_size=20,epochs=15)

y_pred=ann.predict(X_test)
y2_pred=ann.predict(X2_test)

y_pred=list(y_pred)
y2_pred=list(y2_pred)

for i in range(len(y_pred)):
    if y_pred[i] >= .5:
        y_pred[i]=1
    else : y_pred[i]=0

for i in range(len(y2_pred)):
    if y2_pred[i] >= .5:
        y2_pred[i]=1
    else : y2_pred[i]=0

ann.evaluate(X_test,Y_test)
ann.evaluate(X2_test,Y2_test)

history.history
history2.history2

plt.title('Loss / Mean Squared Error w/out Outliers')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Loss / Mean Squared Error w/ Outliers')
plt.plot(history2.history2['loss'], label='train')
plt.plot(history2.history2['val_loss'], label='test')
plt.legend()
plt.show()


print(classification_report(Y_test, y_pred, target_names=["Alarm", "No Alarm"]))
print(classification_report(Y2_test, y2_pred, target_names=["Alarm", "No Alarm"]))

