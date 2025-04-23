![image](https://github.com/user-attachments/assets/bfcd45f3-d2df-446c-8dad-8711e4329ba9)![image](https://github.com/user-attachments/assets/e2d00e13-1ce6-4a5f-b747-e7cead92a038)![image](https://github.com/user-attachments/assets/234cec8b-87e0-4ec6-b732-39c906c0a9ea)![image](https://github.com/user-attachments/assets/6723cdfb-ac49-4433-bec7-6568367a7d73)![image](https://github.com/user-attachments/assets/352ff38b-3603-48e5-981d-e8b0ef523b1f)![image](https://github.com/user-attachments/assets/8b7fb7f8-0d7b-412d-a92e-8baec1b19564)![image](https://github.com/user-attachments/assets/5c05d447-7137-4994-93c2-1d4bba75a230)![image](https://github.com/user-attachments/assets/e8dc8202-bd9a-4637-b4ba-5fbf393fe5a5)![image](https://github.com/user-attachments/assets/7794ee4c-b448-463b-9a9e-207bf1f7eeeb)## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
~~~
![image](https://github.com/user-attachments/assets/4613e232-1e0f-4ee3-9571-e0c3f9bfe321)
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![image](https://github.com/user-attachments/assets/8ccb8a23-b9dd-4b71-8de9-6356e66490f1)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![image](https://github.com/user-attachments/assets/dc9f5727-7d6a-408d-af89-200fb0fee9e7)
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
![image](https://github.com/user-attachments/assets/948362ff-8cfe-45ac-8d33-40e57a944f2d)
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~
![image](https://github.com/user-attachments/assets/48c8bbe6-78b1-404d-b0c2-858887280b29)
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
![image](https://github.com/user-attachments/assets/4b66e459-598c-4fa9-a7e1-890644f64026)
~~~
pip install --upgrade category_encoders
~~~
![image](https://github.com/user-attachments/assets/8585b5d9-58f2-4def-ac0e-e714f126b683)
~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
~~~
![image](https://github.com/user-attachments/assets/942a9e51-a55c-48de-8489-6bc00e4e541a)
~~~
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
~~~
![image](https://github.com/user-attachments/assets/a1cc4e0f-b9f7-40c1-88b9-df6166bd955c)
~~~
dfb=pd.concat([df,nd],axis=1)
dfb
~~~
![image](https://github.com/user-attachments/assets/1d5fbabc-9257-4971-85f7-8487b69965e0)
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
~~~
![image](https://github.com/user-attachments/assets/7dbeb7d9-f935-4588-854e-00eea25f4bb8)
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
~~~
![image](https://github.com/user-attachments/assets/263003ff-fec8-4ada-8770-f17a066e133a)
~~~
df.skew()
~~~
![image](https://github.com/user-attachments/assets/0d58152a-3dbd-4e0e-8fd6-0fb25ccbb870)
~~~
np.log(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/9944e3a4-84d3-4c57-82f1-75c581424e66)
~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/9fa3a316-9b9b-453c-9a78-bf57c26ae8fd)
~~~
np.square(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/7796f76f-db5d-4ce1-9620-7845b29dbe96)
~~~
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/05c9474a-2ccf-40dd-b140-39c701f49888)
~~~
df.skew()
~~~
![image](https://github.com/user-attachments/assets/8f105654-54ad-4311-8617-5bf07da3372b)
~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~
![image](https://github.com/user-attachments/assets/7a374766-bb7b-4803-b43e-9c3690b1bf26)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
![image](https://github.com/user-attachments/assets/18c8c302-e717-43d6-bc76-a73972dd53a8)
~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/7bd0b599-aff0-4b1c-a986-3f2f9c72e632)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/b94da759-1d22-430e-8a38-fbc28b9111a9)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/1f17da26-de42-480b-ac30-29fcbbd4e50a)
~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/6ebffeb5-9aef-46f8-8083-6c0c15a70d7f)
~~~
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
~~~
![image](https://github.com/user-attachments/assets/b3b72d8e-0b8b-47f1-9363-e49b50805ff9)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
~~~
![image](https://github.com/user-attachments/assets/2242fc98-aefc-4c00-80a7-7adfdb45b812)
~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/4adb7bc3-60b1-4a14-a608-f77a7a755b4b)
~~~
# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

