import numpy as np
from sklearn import svm
from data_extraction import *
from sklearn.preprocessing import StandardScaler

def get_mse(x,y):
    return np.mean(np.power((x-y),2))

def construct_svm_reg(X_train, target_train, X_val, target_val, C, kernel, normalize = False, verbose = False):
    svm_reg = svm.SVR(C=C, kernel=kernel)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)
    svm_reg.fit(X_train, target_train)
    if verbose == True: print("SVM regession model fitted.")

    pred_train = svm_reg.predict(X_train)
    train_mse = get_mse(pred_train, target_train)
    if verbose == True: print(f"Training MSE: {train_mse}")
    train_accuracy = np.mean((pred_train>=1).astype(int) == (target_train>=1).astype(int))
    if verbose == True: print(f"Training accuracy: {train_accuracy}")

    pred_val = svm_reg.predict(X_val)
    val_mse = get_mse(pred_val, target_val)
    if verbose == True: print(f"Validation MSE: {val_mse}")
    val_accuracy = np.mean((pred_val>=1).astype(int) == (target_val>=1).astype(int))
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")

    return svm_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy

def construct_svm_class(X_train, target_train, X_val, target_val, C, kernel, normalize = False, verbose = False):
    svm_class = svm.SVC(C=C, kernel=kernel)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)

    svm_class.fit(X_train, target_train)
    if verbose == True: print("SVM classification model fitted.")
    pred_train = svm_class.predict(X_train)
    train_accuracy = np.mean(pred_train == target_train)
    if verbose == True: print(f"Training accuracy: {train_accuracy}")
    pred_val = svm_class.predict(X_val)
    val_accuracy = np.mean(pred_val == target_val)
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")
    print(f"TP: {sum((pred_val==target_val)*(target_val==1))}")
    print(f"FP: {sum((pred_val!=target_val)*(target_val==0))}")
    print(f"FN: {sum((pred_val!=target_val)*(target_val==1))}")
    print(f"TN: {sum((pred_val==target_val)*(target_val==0))}")
    return svm_class, pred_train, train_accuracy, pred_val, val_accuracy

def main():
    X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data("all")
    #construct_svm_reg(X_basic[train_indices, :], target_cont[train_indices],X_basic[val_indices], target_cont[val_indices], C=0.1, kernel = "rbf", verbose = True, normalize = True)
    print("best overall")
    construct_svm_class(X_technical[train_indices, :], target_bin[train_indices],X_technical[val_indices], target_bin[val_indices], C=0.1, kernel = "linear", verbose = True, normalize = False)
    print("best basic features")
    construct_svm_class(X_basic[train_indices, :], target_bin[train_indices],X_basic[val_indices], target_bin[val_indices], C=0.1, kernel = "rbf", verbose = True, normalize = True)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
