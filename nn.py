import numpy as np
from data_extraction import *
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
def get_mse(x,y):
    return np.mean(np.power((x-y),2))

def construct_nn_reg(X_train, target_train, X_val, target_val, activation, alpha, normalize = False, verbose = False):
    hidden_layer_sizes = (3 * min(X_train.shape))
    nn_reg = MLPRegressor(hidden_layer_sizes, activation = activation, solver = "adam", alpha=alpha, max_iter = 2000, random_state = 0)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)
    nn_reg.fit(X_train, target_train)
    if verbose == True: print("nn regession model fitted.")

    pred_train = nn_reg.predict(X_train)
    train_mse = get_mse(pred_train, target_train)
    if verbose == True: print(f"Training MSE: {train_mse}")
    train_accuracy = np.mean((pred_train>=1).astype(int) == (target_train>=1).astype(int))
    if verbose == True: print(f"Training accuracy: {train_accuracy}")

    pred_val = nn_reg.predict(X_val)
    val_mse = get_mse(pred_val, target_val)
    if verbose == True: print(f"Validation MSE: {val_mse}")
    val_accuracy = np.mean((pred_val>=1).astype(int) == (target_val>=1).astype(int))
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")

    return nn_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy

def construct_nn_class(X_train, target_train, X_val, target_val, activation, alpha, normalize = False, verbose = False):
    hidden_layer_sizes = (3* min(X_train.shape),)
    nn_class = MLPClassifier(hidden_layer_sizes, activation = activation, solver = "adam", alpha=alpha, max_iter = 2000, random_state = 0)

    if normalize == True:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_val = x_scaler.fit_transform(X_val)

    nn_class.fit(X_train, target_train)
    if verbose == True: print("nn classification model fitted.")
    pred_train = nn_class.predict(X_train)
    train_accuracy = np.mean(pred_train == target_train)
    if verbose == True: print(f"Training accuracy: {train_accuracy}")
    pred_val = nn_class.predict(X_val)
    val_accuracy = np.mean(pred_val == target_val)
    if verbose == True: print(f"Validation accuracy: {val_accuracy}")

    return nn_class, pred_train, train_accuracy, pred_val, val_accuracy
def make_train_plot(X_basic, target_bin, train_indices, val_indices):
    hidden_layer_sizes = (3* min(X_basic.shape),)
    nn_class = MLPClassifier(hidden_layer_sizes, activation = "relu", solver = "adam", alpha=0.001, max_iter = 2000, random_state = 0)
    train_sizes, train_scores, valid_scores = learning_curve(nn_class, X_basic,target_bin, train_sizes=[270],cv= [(train_indices, val_indices)])
    pdb.set_trace()
    plt.clf()
    epochs = np.linspace(1,train_scores.shape[0], train_scores.shape[0])

def main():
    X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data("all")
    #construct_nn_reg(X_basic[train_indices, :], target_cont[train_indices],X_basic[val_indices], target_cont[val_indices], activation="relu", alpha = 0.0001, verbose = True, normalize = True)
    #construct_nn_class(X_basic[train_indices, :], target_bin[train_indices],X_basic[val_indices], target_bin[val_indices], activation="relu", alpha = 0.0001, verbose = True, normalize = True)
    make_train_plot(X_basic, target_bin,train_indices, val_indices)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
