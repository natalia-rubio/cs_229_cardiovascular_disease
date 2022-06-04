from data_extraction import *
from pca import *

def run_pca_parameter_search(normalize, feature_type, filename):
    feature_list = ["basic", "technical"]
    dataset_list = ["cleveland", "hungarian", "long_beach", "switzerland", "all"]
    n_components_opt = [1,2,3,4,5,6]
    reg_results = {}
    results_arr = np.empty((5,13), dtype = object)
    for inst_ind in range(len(dataset_list)):
        inst_name = dataset_list[inst_ind]
        X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data(inst_name)
        print(dataset_list[inst_ind] + "num train samples " + str(train_indices.size) + "num validation samples " + str(val_indices.size))
        results_arr[inst_ind,0] = inst_name
        if feature_type == "basic":
            X = X_basic # use only basic features
        else:
            X = X_technical
        for n_components_ind in range(len(n_components_opt)):

                print(f"Fitting pca classification model on {inst_name} {feature_type} data with {n_components_opt[n_components_ind]} n_components .")
                lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy = \
                construct_pca_reg(X[train_indices, :], target_cont[train_indices],X[val_indices], target_cont[val_indices], \
                n_components = n_components_opt[n_components_ind], verbose = True, normalize = normalize)
                results_arr[inst_ind, 1+n_components_ind] = str(val_accuracy)[0:4]

                print(f"Fitting pca classification model on {inst_name} {feature_type} data with {n_components_opt[n_components_ind]} n_components .")
                lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy = \
                construct_pca_class(X[train_indices, :], target_bin[train_indices],X[val_indices], target_bin[val_indices], \
                n_components = n_components_opt[n_components_ind], verbose = True, normalize = normalize)
                results_arr[inst_ind, 7+n_components_ind] = str(val_accuracy)[0:4]

    np.savetxt(filename, results_arr, delimiter=' & ', fmt = '%s', newline=' \\\\\n')
# print("Upcaormalized, Basic Features")
# run_pca_parameter_search(False, feature_type = "basic", filename = "log_lin_val_table_basic.csv")
print("Normalized, Basic Features")
run_pca_parameter_search(True, feature_type = "basic", filename = "pca_val_table_norm_basic.csv")
run_pca_parameter_search(False, feature_type = "basic", filename = "pca_val_table_basic.csv")
# print("Upcaormalized, Technical Features")
# run_pca_parameter_search(False, feature_type = "technical", filename = "log_lin_val_table_technical.csv")
print("Normalized, Technical Features")
run_pca_parameter_search(True, feature_type = "technical", filename = "pca_val_table_norm_technical.csv")
run_pca_parameter_search(False, feature_type = "technical", filename = "pca_val_table_technical.csv")
