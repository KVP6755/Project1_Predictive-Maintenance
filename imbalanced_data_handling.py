#week 2: HANDLING DATA IMBALANCE
#Sets up oversampling (SMOTE) and class-weighting approaches to handle the severe class imbalance in machine failure prediction (~3.4% failures).

#AUTHOR: VARNIKA VALLIAMMAI V

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data_path=r"C:\Users\varni\OneDrive\Desktop\data\ai4i_rolling_features.csv"
df = pd.read_csv(data_path)
print("loaded shape: ",df.shape)

leakage_cols=['TWF','HDF','PWF','OSF','RNF']
id_cols=['UDI','Product ID','date']
target='Machine failure'

X=df.drop(columns=leakage_cols+id_cols + [target])
X=pd.get_dummies(X,columns=['Type'],drop_first=True)
y=df[target]

print("Feature shape: ",X.shape)
print("target distribution: ",y.value_counts())

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2, stratify=y, random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train target distribution:\n", y_train.value_counts())

sm=SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE, train target distribution:\n", y_train_smote.value_counts())
