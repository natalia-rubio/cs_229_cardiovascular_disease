import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from data_extraction import *
from sklearn.preprocessing import StandardScaler

def get_mse(x,y):
    return np.mean(np.power((x-y),2))

def construct_lin_reg(X_train, target_train, X_val, target_val, reg, normalize = False, verbose = False):
    if reg == 0:
        lin_reg = LinearRegression()
    elif reg == 1:
        lin_reg = Lasso()
    elif reg == 2:
        lin_reg = Ridge()

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)

    lin_reg.fit(X_train, target_train)
    if verbose == True: print("Linear regession model fitted.")
    pred_train = lin_reg.predict(X_train)
    train_mse = get_mse(pred_train, target_train)
    if verbose == True: print(f"Training MSE: {train_mse}")
    train_accuracy = np.mean((pred_train>=1).astype(int) == (target_train>=1).astype(int))
    if verbose == True: print(f"Training accuracy: {train_accuracy}")

    pred_val = lin_reg.predict(X_val)
    val_mse = get_mse(pred_val, target_val)
    if verbose == True: print(f"Validation MSE: {val_mse}")
    val_accuracy = np.mean((pred_val>=1).astype(int) == (target_val>=1).astype(int))
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")
    return lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy

def main():
    X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data("cleveland")
    construct_lin_reg(X_basic[train_indices, :], target_cont[train_indices],X_basic[val_indices], target_cont[val_indices], reg = 0, verbose = True)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
