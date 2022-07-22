import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv('Iris.csv')
df.head(6)

#dropping the id column

df = df.drop(columns=['Id'])

# To display the status about data
p=df.describe()

#To get the labels datatypes 
q=df.info()

#To display the no of samples on each class
df['Species'].value_counts()

#Checking for null values
df.isnull().sum()

df['SepalLengthCm'].hist()

df['SepalWidthCm'].hist()

df['PetalLengthCm'].hist()

df['PetalWidthCm'].hist()




k=df.corr()
figure,ax=plt.subplots(figsize=(5,4))
sns.heatmap(k,annot=True,ax =ax,cmap='coolwarm')
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#df['Species'] = le.fit_transform(df['Species'])
df.head()
x=df.drop(columns=['Species'])
y=df['Species']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# model training
model.fit(x_train, y_train)

# print metric to get performance
result=model.score(x_test, y_test) * 100

# save the model
import pickle
filename = 'savedmodel.sav'
pickle.dump(model, open(filename, 'wb'))
load_model = pickle.load(open(filename,'rb'))
load_model.predict([[6.0, 2.2, 4.0, 1.0]])






