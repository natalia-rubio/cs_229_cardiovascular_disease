import numpy as np
from data_extraction import *
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def get_mse(x,y):
    return np.mean(np.power((x-y),2))

def construct_gauss_mix_class(X_train, target_train, X_val, target_val, n_components, normalize = False, verbose = False):

    gauss_mix_class = GaussianMixture(n_components, covariance_type = "tied", random_state = 0)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)

    gauss_mix_class.fit(X_train, target_train)
    if verbose == True: print("gauss_mix classification model fitted.")
    pred_train = gauss_mix_class.predict(X_train)
    train_accuracy = np.mean(pred_train == target_train)
    if train_accuracy < 0.5:
        train_accuracy = 1-train_accuracy
    if verbose == True: print(f"Training accuracy: {train_accuracy}")
    pred_val = gauss_mix_class.predict(X_val)
    val_accuracy = np.mean(pred_val == target_val)
    if val_accuracy < 0.5:
        val_accuracy = 1-val_accuracy
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")

    return gauss_mix_class, pred_train, train_accuracy, pred_val, val_accuracy

def main():
    X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data("cleveland")
    construct_gauss_mix_class(X_basic[train_indices, :], target_bin[train_indices],X_basic[val_indices], target_bin[val_indices], n_components = 6, verbose = True, normalize = True)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
