from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

import warnings
import pickle

import numpy as np
import pandas as pd


threshold = 0.5

df = pd.read_parquet("prepared_clinvar_conflicting.parquet")
print(df.head(n=10))

target_variable = "class"
print(f"Target variable: '{target_variable}'")

# Identify numerical features after type conversion
numerical_features = df.select_dtypes(include=np.number).columns
if target_variable in numerical_features:
    numerical_features = numerical_features.drop(target_variable)
print(f"Numerical features: {sorted(numerical_features.tolist())}")

# Scaling numerical values
mm = MinMaxScaler()
for column in [numerical_features]:
    df[column] = mm.fit_transform(df[column])

# Drop correlated variables
correlated_columns = ["af_exac", "af_tgp", "cdna_position", "cds_position"]

# Re-apply the numerical features filtering with .difference()
numerical_features = numerical_features.difference(correlated_columns)

df.drop(columns=correlated_columns, inplace=True)

# Identify categorical features after type conversion
categorical_features = df.select_dtypes(include='category').columns
if target_variable in categorical_features:
    categorical_features = categorical_features.drop(target_variable)
print(f"Categorical features: {sorted(categorical_features.tolist())}")


# Correctly combine categorical and numerical features by converting to lists first
all_features = categorical_features.tolist() + numerical_features.tolist()


common_y_name = target_variable
common_random_state = 11562788

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=common_random_state
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=common_random_state
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Length of the train dataset: {len(df_train)}")
print(f"Length of the validation dataset: {len(df_val)}")
print(f"Length of the test dataset: {len(df_test)}")

def split_y_X(df, y_name):
    y = df[y_name]
    X = df.drop(columns=[y_name])
    return y, X


y_full_train, X_full_train = split_y_X(df_full_train, common_y_name)
y_train, Xtmp_train = split_y_X(df_train, common_y_name)
y_val, Xtmp_val = split_y_X(df_val, common_y_name)
y_test, Xtmp_test = split_y_X(df_test, common_y_name)


dv = DictVectorizer(sparse=False)

train_dict = Xtmp_train.to_dict(orient="records")
X_train = dv.fit_transform(train_dict)
print(f"Training features: {X_train.shape}")

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
model.fit(X_train, y_train)
print(f"Model: {model}")

val_dict = Xtmp_val.to_dict(orient="records")
X_val = dv.transform(val_dict)
print(f"Validation features: {X_val.shape}")

y_pred = model.predict_proba(X_val)[:, 1]

roc_auc = round(roc_auc_score(y_val, y_pred >= threshold), 3)
print(f"AUC: {roc_auc}")