#week 2: HANDLING DATA IMBALANCE
#Sets up oversampling (SMOTE) and class-weighting approaches to handle the severe class imbalance in machine failure prediction (~3.4% failures).

#AUTHOR: VARNIKA VALLIAMMAI V

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


#1. Load dataset
data_path=r"C:\Users\varni\OneDrive\Desktop\data\ai4i_rolling_features.csv"
df = pd.read_csv(data_path)
print("loaded shape: ",df.shape)


#2. Prepare features and target
leakage_cols=['TWF','HDF','PWF','OSF','RNF']
id_cols=['UDI','Product ID','date']
target='Machine failure'

X=df.drop(columns=leakage_cols+id_cols + [target])
X=pd.get_dummies(X,columns=['Type'],drop_first=True)
y=df[target]

print("Feature shape: ",X.shape)
print("target distribution: ",y.value_counts())


# 3. Train/test split — BEFORE any oversampling
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2, stratify=y, random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train target distribution:\n", y_train.value_counts())


#4. Apply SMOTE — only on training data
sm=SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE, train target distribution:\n", y_train_smote.value_counts())


# 5.ALTERNATIVE APPROACH

from sklearn.ensemble import RandomForestClassifier

model_balanced= RandomForestClassifier(
    class_weight='balanced',
    random_state = 42
)

model_balanced.fit(X_train,y_train)
print("Train accuracy:", model_balanced.score(X_train, y_train))
print("Test accuracy :", model_balanced.score(X_test, y_test))


# 6. Quick comparison helper
def compare_class_distribution(y_before, y_after, label_before, label_after):
    """Prints a side-by-side count comparison of two target distributions."""
    print(f"\n{label_before}:")
    print(y_before.value_counts())
    print(f"\n{label_after}:")
    print(y_after.value_counts())

compare_class_distribution(
    y_train, y_train_smote,
    "Original training distribution",
    "After SMOTE distribution"
)


# 7. Precision-Recall curve plotting function
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall(model, X_test, y_test,label="model"):
    """
     Plots the Precision-Recall curve for a fitted model.
    Use this instead of accuracy — accuracy is misleading on
    imbalanced data (a model predicting "no failure" always
    would score ~96% accuracy while being useless).
    """

    probs = model.predict_proba(X_test)[:, 1]  # probability of "failure"
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    avg_precision = average_precision_score(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'{label} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall (failures caught)')
    plt.ylabel('Precision (alarms that were real)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'pr_curve_{label}.png')
    plt.show()

    return precision, recall, thresholds
# Test the function works using our Day 2 model
plot_precision_recall(model_balanced, X_test, y_test, label="balanced_rf_test")

#8. ROC-AUC curve plotting function

from sklearn.metrics import roc_curve, auc

def plot_roc_auc(model, X_test, y_test, label="model"):
    """
    Plots the ROC curve and calculates AUC (Area Under Curve).
    AUC = 0.5 means random guessing; AUC = 1.0 means perfect separation.
    """
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'roc_curve_{label}.png')
    plt.show()

    return fpr, tpr, roc_auc

# Test the function works using our Day 2 model
plot_roc_auc(model_balanced, X_test, y_test, label="balanced_rf_test")

# 9. Summary — what this script provides to the team
print("\n" + "=" * 50)
print("TRACK 3 DELIVERABLES READY:")
print("=" * 50)
print("1. SMOTE-resampled training data: X_train_smote, y_train_smote")
print("2. class_weight='balanced' model setup: model_balanced")
print("3. plot_precision_recall(model, X_test, y_test, label) function")
print("4. plot_roc_auc(model, X_test, y_test, label) function")
