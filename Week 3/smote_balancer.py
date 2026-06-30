
"""
Week 3 - Imbalanced Classification

Member 2: Imbalance Data Expert

Author:
Shilpa S Nair

Description:
Reusable SMOTE balancing functions for handling class imbalance.

These functions are designed to integrate with the
5-Fold Stratified Cross Validation pipeline.
"""

# Import libraries
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

#Apply standard SMOTE to the training data.
def apply_smote(X_train, y_train, random_state=42):
    
    smote = SMOTE(random_state=random_state)

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled



# Apply BorderlineSMOTE to the training data.

def apply_borderline_smote(X_train, y_train, random_state=42):

    borderline = BorderlineSMOTE(random_state=random_state)

    X_resampled, y_resampled = borderline.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


# Print class distributions before and after balancing.
def compare_class_distribution(y_before, y_after):

    print("=" * 40)
    print("Class Distribution Comparison")
    print("=" * 40)

    print("\nBefore Balancing:")
    print(y_before.value_counts())

    print("\nAfter Balancing:")
    print(y_after.value_counts())



#Return class counts.
def get_class_counts(y):
    return y.value_counts()

if __name__ == "__main__":
    print("SMOTE Balancer Module Loaded Successfully!")