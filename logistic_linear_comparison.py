from data_extraction import *
from logistic_regression import *
from linear_regression import *

def run_log_lin_comp(normalize, filename):
    feature_list = ["basic", "technical"]
    dataset_list = ["cleveland", "hungarian", "long_beach", "switzerland", "all"]
    reg_opt = [0, 1, 2]
    reg_results = {}
    results_arr = np.empty((5,13), dtype = object)
    for i in range(len(dataset_list)):
        inst_name = dataset_list[i]
        X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data(inst_name)
        results_arr[i,0] = inst_name
        for feature_type in feature_list:
            if feature_type == "basic":
                X = X_basic; feat_ind = 0
            else:
                X = X_technical; feat_ind = 1
            for reg in reg_opt:
                print(f"Fitting linear regression model on {inst_name} {feature_type} data with {reg} regularization.")
                lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy = \
                construct_lin_reg(X[train_indices, :], target_cont[train_indices],X[val_indices], target_cont[val_indices], normalize, reg)
                results_arr[i, 1+3*feat_ind+reg] = str(val_accuracy)[0:4]

                print(f"Fitting logistic regression model on {inst_name} {feature_type} data with {reg} regularization.")
                log_reg, pred_train, train_accuracy, pred_val, val_accuracy = \
                construct_log_reg(X_basic[train_indices, :], target_bin[train_indices],X_basic[val_indices], target_bin[val_indices], normalize, reg)
                results_arr[i, 7+3*feat_ind+reg] = str(val_accuracy)[0:4]

    np.savetxt(filename, results_arr, delimiter=' & ', fmt = '%s', newline=' \\\\\n')
print("Unnormalized")
run_log_lin_comp(False, "log_lin_val_table.csv")
print("Normalized")
run_log_lin_comp(True, "log_lin_val_table_norm.csv")
