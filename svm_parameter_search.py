from data_extraction import *
from svm import *

def run_svm_parameter_search(normalize, feature_type, filename):
    feature_list = ["basic", "technical"]
    dataset_list = ["cleveland", "hungarian", "long_beach", "switzerland", "all"]
    kernel_opt = ["linear", "rbf"]
    C_opt = [0.1,1,10]
    reg_results = {}
    results_arr = np.empty((5,13), dtype = object)
    for inst_ind in range(len(dataset_list)):
        inst_name = dataset_list[inst_ind]
        X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data(inst_name)
        results_arr[inst_ind,0] = inst_name
        if feature_type == "basic":
            X = X_basic # use only basic features
        else:
            X = X_technical
        for kernel_ind in range(len(kernel_opt)):
            for C_ind in range(len(C_opt)):
                print(f"Fitting SVM regression model on {inst_name} {feature_type} data with {kernel_opt[kernel_ind]} kernel and degree {C_opt[C_ind]}.")
                lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy = \
                construct_svm_reg(X[train_indices, :], target_cont[train_indices],X[val_indices], target_cont[val_indices], \
                C=C_opt[C_ind], kernel = kernel_opt[kernel_ind], verbose = True, normalize = normalize)
                results_arr[inst_ind, 1+3*kernel_ind+C_ind] = str(val_accuracy)[0:4]

                print(f"Fitting SVM classification model on {inst_name} {feature_type} data with {kernel_opt[kernel_ind]} kernel and degree {C_opt[C_ind]}.")
                log_reg, pred_train, train_accuracy, pred_val, val_accuracy = \
                construct_svm_class(X[train_indices, :], target_bin[train_indices],X[val_indices], target_bin[val_indices], \
                C=C_opt[C_ind], kernel = kernel_opt[kernel_ind], verbose = True, normalize = normalize)
                results_arr[inst_ind, 7+3*kernel_ind+C_ind] = str(val_accuracy)[0:4]

    np.savetxt(filename, results_arr, delimiter=' & ', fmt = '%s', newline=' \\\\\n')
print("Unnormalized, Basic Features")
run_svm_parameter_search(False, feature_type = "basic", filename = "log_lin_val_table_basic.csv")
print("Normalized, Basic Features")
run_svm_parameter_search(True, feature_type = "basic", filename = "log_lin_val_table_norm_basic.csv")
print("Unnormalized, Technical Features")
run_svm_parameter_search(False, feature_type = "technical", filename = "log_lin_val_table_technical.csv")
print("Normalized, Technical Features")
run_svm_parameter_search(True, feature_type = "technical", filename = "log_lin_val_table_norm_technical.csv")
