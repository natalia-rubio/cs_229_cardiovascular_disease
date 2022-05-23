import numpy as np
from sklearn.linear_model import LogisticRegression
from data_extraction import *
from sklearn.preprocessing import StandardScaler

def construct_log_reg(X_train, target_train, X_val, target_val, reg, normalize = False, verbose = False):
    if reg == 0:
        log_reg = LogisticRegression(solver = "liblinear")
    elif reg == 1:
        log_reg = LogisticRegression(penalty = "l1", solver = "liblinear")
    elif reg == 2:
        log_reg = LogisticRegression(penalty = "l2", solver = "liblinear")

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)

    log_reg.fit(X_train, target_train)
    if verbose == True: print("Logistic regession model fitted.")
    pred_train = log_reg.predict(X_train)
    train_accuracy = np.mean(pred_train == target_train)
    if verbose == True: print(f"Training accuracy: {train_accuracy}")
    pred_val = log_reg.predict(X_val)
    val_accuracy = np.mean(pred_val == target_val)
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")
    return log_reg, pred_train, train_accuracy, pred_val, val_accuracy

def main():
    X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data("cleveland")
    construct_log_reg(X_basic[train_indices, :], target_bin[train_indices],X_basic[val_indices], target_bin[val_indices], verbose = True)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
