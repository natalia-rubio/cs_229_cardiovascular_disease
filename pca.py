import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from data_extraction import *
from sklearn.preprocessing import StandardScaler

def get_mse(x,y):
    return np.mean(np.power((x-y),2))

def construct_pca_reg(X_train, target_train, X_val, target_val, n_components, normalize = False, verbose = False):
    lin_reg = LinearRegression()
    pca = PCA(n_components)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)
    X_train_pca = pca.fit_transform(X_train)
    lin_reg.fit(X_train_pca, target_train)
    if verbose == True: print("Linear regession model fitted.")
    pred_train = lin_reg.predict(X_train_pca)
    train_mse = get_mse(pred_train, target_train)
    if verbose == True: print(f"Training MSE: {train_mse}")
    train_accuracy = np.mean((pred_train>=1).astype(int) == (target_train>=1).astype(int))
    invert = False

    if verbose == True: print(f"Training accuracy: {train_accuracy}")

    X_val_pca = pca.transform(X_val)
    pred_val = lin_reg.predict(X_val_pca)
    val_mse = get_mse(pred_val, target_val)
    if verbose == True: print(f"Validation MSE: {val_mse}")
    val_accuracy = np.mean((pred_val>=1).astype(int) == (target_val>=1).astype(int))
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")
    return lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy

def construct_pca_class(X_train, target_train, X_val, target_val, n_components, normalize = False, verbose = False):
    log_reg = LogisticRegression()
    pca = PCA(n_components)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)
    X_train_pca = pca.fit_transform(X_train)
    log_reg.fit(X_train_pca, target_train)
    if verbose == True: print("Logistic regession model fitted.")
    pred_train = log_reg.predict(X_train_pca)
    train_mse = get_mse(pred_train, target_train)
    if verbose == True: print(f"Training MSE: {train_mse}")
    train_accuracy = np.mean(pred_train == target_train)
    invert = False

    if verbose == True: print(f"Training accuracy: {train_accuracy}")

    X_val_pca = pca.transform(X_val)
    pred_val = log_reg.predict(X_val_pca)
    val_mse = get_mse(pred_val, target_val)
    if verbose == True: print(f"Validation MSE: {val_mse}")
    val_accuracy = np.mean(pred_val == target_val)
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")
    return log_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy

def main():
    X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data("all")
    construct_pca_reg(X_basic[train_indices, :], target_cont[train_indices],X_basic[val_indices], target_cont[val_indices], n_components = 2, normalize = True, verbose = True)
    construct_pca_class(X_basic[train_indices, :], target_bin[train_indices],X_basic[val_indices], target_bin[val_indices], n_components = 2, normalize = True, verbose = True)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
