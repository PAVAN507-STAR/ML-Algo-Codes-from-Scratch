import numpy as np;
import pandas as pd;


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

def bestlambda(X,Y,k,lambdas,lr,epochs):
    if Y.ndim==1:
        Y.reshape(-1,1)
    
    m=X.shape[0]
    
    #Random permuatations or indexed X,Y for better results
    indices = np.random.permutation(m)
    X_shuf=X[indices]
    Y_shuf= Y[indices]

    #Split data into k folds
    X_folds = np.array_split(X_shuf,k)
    Y_folds = np.array_split(Y_shuf,k)

    results = []
    
    for lambda_ in lambdas:
        mse_list=[]

        for i in range(k):
            #test or validation folds
            val_x = X_folds[i]
            val_y=Y_folds[i]

            #training folds
            train_ind = [j for j in range(k) if i!=j]
            train_x=np.vstack([X_folds[j]  for j in train_ind])
            train_y=np.vstack([Y_folds[j]  for j in train_ind])

            theta = np.random.randn(X.shape[1],1)

            for epoch in range(epochs):
                for _ in range(m):
                    idx = np.random.randint(train_x.shape[0])
                    xi = X[idx:idx+1]
                    yi = Y[idx:idx+1]
                    
                    error = xi @ theta - yi
                    gradient = 2 * xi.T @ error + 2 * lambda_ * theta
                    theta -= lr * gradient
        
            pred_Y = validation_X @ theta
            mse = np.mean((pred_Y - validation_Y) ** 2)
            mse_list.append(mse)

        #mean avg error 
        avg_mse = np.mean(mse_list)
        results.append((lambda_, avg_mse))
    results.sort(key=lambda x: x[1] )

    print(results)
    return results[0][0]

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


lambda_ = bestlambda(X,Y,5,[0.001, 0.01, 0.1, 1, 10],0.01,1000)


print("best lambda:",lambda_)






        
    
