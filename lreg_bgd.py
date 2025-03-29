import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

def preprocessdata(df):
    df = df.copy()
    cat_groups = {}
    dummy_dfs = []
    for colname, coldata in df.items():
        #One-Hot encoding shitt
        if colname != 'Rent' and coldata.dtype == 'object':
            uvals = coldata.unique()
            dummy_list = []
            for val in uvals:
                dummy = pd.Series((coldata == val).astype(int), name=f"{colname}_{val}")
                dummy_list.append(dummy)
            if dummy_list:
                dummy_df = pd.concat(dummy_list, axis=1)
                dummy_dfs.append(dummy_df)
                cat_groups[colname] = dummy_df.columns.tolist()
            #Remove old cols
            df.drop(columns=[colname], inplace=True)
    if dummy_dfs:
        df = pd.concat([df] + dummy_dfs, axis=1)
        df = df.copy()
    return df, cat_groups

t0,t1=5,50

#used to choose learning rate for SGD dynamically
def lr_scheduler(t):
    return t0/(t+t1)


def RidgeReg(X,Y,tx,lambda_):
    lr = 0.01  # Fixed learning rate
    epochs = 1000  # More epochs
    m = Y.shape[0]
    theta = np.random.randn(X.shape[1], 1)
    for epoch in range(epochs):
        for i in range(m):
            randomindex = np.random.randint(m)
            xi = X[randomindex:randomindex+1]
            yi = Y[randomindex:randomindex+1]
            gradient = 2 *(xi @ theta - yi)@ xi.T+ 2*lambda_*theta
            theta = theta - lr * gradient
    print("Final Theta:", theta)
    ty = tx @ theta
    return ty


#Linear Reg using stochastic gradient 
def SGD(X, Y, tx):
    lr = 0.01  # Fixed learning rate
    epochs = 1000  # More epochs
    m = Y.shape[0]
    theta = np.random.randn(X.shape[1], 1)
    for epoch in range(epochs):
        for i in range(m):
            randomindex = np.random.randint(m)
            xi = X[randomindex:randomindex+1]
            yi = Y[randomindex:randomindex+1]
            gradient = 2 *(xi @ theta - yi)@xi.T
            theta = theta - lr * gradient
    print("Final Theta:", theta)
    ty = tx @ theta
    return ty

def linearRegression(X, Y, tx):
    lr = 0.01
    m = X.shape[0]
    iterations = 5000
    theta = np.random.randn(X.shape[1], 1)
    for i in range(iterations):
        prediction = X @ theta
        error = prediction - Y
        gradient = (2/m) * X.T @ error  # Gradient calculation
        theta = theta - lr * gradient   # Theta update
        loss = (1/m) * np.sum(error ** 2)
        if i % 500 == 0:  # Optional: print progress
            print(f"Iteration {i}, Loss: {loss}")
    print("Final theta:", theta)
    ty = tx @ theta
    return ty
    
df = pd.read_csv('House_Rent_Dataset.csv')
df.drop(columns=['Posted On'], inplace=True)
df = df.dropna()

Y = df['Rent'].values.reshape(-1, 1).astype(float)
X_df, cat_groups = preprocessdata(df)

if 'Rent' in X_df.columns:
    X_df.drop(columns=['Rent'], inplace=True)

X_df = X_df.astype(float)

dummy_cols = [col for group in cat_groups.values() for col in group]
numeric_cols = [col for col in X_df.columns if col not in dummy_cols]
means = {col: X_df[col].mean() for col in numeric_cols}
stds = {col: X_df[col].std() for col in numeric_cols}

for col in numeric_cols:
    X_df[col] = (X_df[col] - means[col]) / stds[col]

X_df['bias'] = 1
X = X_df.values

user_inputs = {}

for col in numeric_cols:
    val = float(input(f"Enter value for {col}: "))
    user_inputs[col] = (val - means[col]) / stds[col]

for group, cols in cat_groups.items():
    options = [col.split('_')[1] for col in cols]
    print(f"For {group}, choose one:")
    selected = input("Enter your choice: ").strip()
    for col in cols:
        user_inputs[col] = 1 if col.split('_')[1] == selected else 0

user_inputs['bias'] = 1
tx = np.array([user_inputs[col] for col in X_df.columns]).reshape(1, -1)

ty = linearRegression(X, Y, tx)
print("Predicted value from lreg_BGD:", ty)

ty=SGD(X,Y,tx)
print("Predicted value from lreg_SGD:", ty)

#==Scikit Learn ==*
df2 = pd.read_csv('House_Rent_Dataset.csv')
df2.drop(columns=['Posted On'], inplace=True)
df2 = df2.dropna()

Y2 = df2['Rent'].values.reshape(-1, 1).astype(float)
X2 = df2.drop(columns=['Rent'])

cat_cols = X2.select_dtypes(include=['object']).columns.tolist()
num_cols = X2.select_dtypes(exclude=['object']).columns.tolist()
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
model =([
    ('preprocessing', preprocessor),
    ('regressor', SGDRegressor())
])
model.fit(X2, Y2)
user_inputs_sklearn = {}

for col in num_cols:
    user_inputs_sklearn[col] = float(input(f"Enter value for {col}: "))

for col in cat_cols:
    options = df2[col].unique()
    print(f"For {col}, choose one: ")
    user_inputs_sklearn[col] = input("Enter your choice: ").strip()

user_df = pd.DataFrame([user_inputs_sklearn])
pred = model.predict(user_df)
print(pred)
