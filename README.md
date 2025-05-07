# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")

df.head()
df.dropna()
```
![image](https://github.com/user-attachments/assets/75ecad83-1c00-4428-8d9f-9c757865d8f5)

```
max_vals=np.max(np.abs(df[['Height']]))
max_vals
max_vals1=np.max(np.abs(df[['Weight']]))
max_vals1
print("Height =",max_vals)
print("Weight =",max_vals1)
```
![image](https://github.com/user-attachments/assets/6af30d0b-373c-46cf-940c-314756295f06)

```
df1=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/13e4a188-c3b2-4229-97b3-555037c682b9)
```

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/7ec71488-c8e4-4b7d-8cb7-425abe4842f8)

```
from sklearn.preprocessing import Normalizer
df2=pd.read_csv("/content/bmi.csv")
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/60737c08-e02f-42e8-85cb-2b87bad3169a)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/ba1850f4-7d1d-4a79-90c2-8a5666604bf7)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/24a84400-8b4f-42df-99c7-2956c9e91132)
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/55a8d465-7917-4cd6-841d-442b539e4475)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/683f097a-a27d-4cd4-bb73-08c5594f9ae9)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/85c4a3fe-10d4-4886-b3fa-552623899c75)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/ffac76b7-f3dc-4b60-9fb9-eb8f81885061)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/a6a74326-5427-4ebe-b20d-02a80db8f94f)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/9165f6f7-40d6-46a6-b96d-01f7b5bcf837)
```
data2
```
![image](https://github.com/user-attachments/assets/53fdf7af-4dd9-4edf-b440-d09475bd50d3)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/719340d7-2f64-46f3-b730-29e8e1949e1a)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/fab1e847-d345-4667-8909-c10ccadce17d)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/d64ab6d6-19b8-4586-bf3b-871bec9a46e4)
```
y=new_data['SalStat']
print(y)
```
![image](https://github.com/user-attachments/assets/9305927c-ed01-4537-b41f-a368721f346c)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/9736129f-6def-4d44-9907-85849fd32f4f)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/7dec858b-2193-45f2-9566-84a6cfe4c10c)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/c0453ca3-0343-4aab-9eab-e6ed4ed7cd41)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)

X=df[['Feature1','Feature3']]
y=df['Target']

selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

print("Selected Features:", selected_features)
```
![image](https://github.com/user-attachments/assets/2c05e284-c3b8-4d7f-a3ef-6aeafbf9460a)

# RESULT:
  Thus, Feature selection and Feature scaling has been used on the given dataset.
