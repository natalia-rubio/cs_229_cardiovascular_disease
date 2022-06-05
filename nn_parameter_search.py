from data_extraction import *
from nn import *
#import matplotlib
import matplotlib.pyplot as plt
def run_nn_parameter_search(normalize, feature_type, type, filename):
    feature_list = ["basic", "technical"]
    norm_list = [False, True]
    #dataset_list = ["cleveland", "hungarian", "long_beach", "switzerland", "all"]
    dataset_list = ["all"]
    activation_opt = ["identity", "relu", "logistic"]
    alpha_opt = [10,0.01,0.00001]
    reg_results = {}
    results_arr = np.empty((5,10), dtype = object)
    train_accuracy_list = []
    val_accuracy_list = []
    for inst_ind in range(len(dataset_list)):
        inst_name = dataset_list[inst_ind]
        X_basic, X_technical, target_bin, target_cont, train_indices, val_indices = load_data(inst_name)
        results_arr[inst_ind,0] = inst_name
        for feature_type in feature_list:
            for normalize in norm_list:
                if feature_type == "basic":
                    X = X_basic # use only basic features
                else:
                    X = X_technical
                for activation_ind in range(len(activation_opt)):
                    for alpha_ind in range(len(alpha_opt)):
                        if type == "reg":
                            print(f"Fitting NN regression model on {inst_name} {feature_type} data with {activation_opt[activation_ind]} activation and degree {alpha_opt[alpha_ind]}.")
                            lin_reg, pred_train, train_accuracy, train_mse, pred_val, val_mse, val_accuracy = \
                            construct_nn_reg(X[train_indices, :], target_cont[train_indices],X[val_indices], target_cont[val_indices], \
                            alpha=alpha_opt[alpha_ind], activation = activation_opt[activation_ind], verbose = True, normalize = normalize)
                            results_arr[inst_ind, 1+3*activation_ind+alpha_ind] = str(val_accuracy)[0:4]
                            train_accuracy_list.append(train_accuracy); val_accuracy_list.append(val_accuracy)
                        if type == "class":
                            print(f"Fitting nn classification model on {inst_name} {feature_type} data with {activation_opt[activation_ind]} activation and degree {alpha_opt[alpha_ind]}.")
                            log_reg, pred_train, train_accuracy, pred_val, val_accuracy = \
                            construct_nn_class(X[train_indices, :], target_bin[train_indices],X[val_indices], target_bin[val_indices], \
                            alpha=alpha_opt[alpha_ind], activation = activation_opt[activation_ind], verbose = True, normalize = normalize)
                            results_arr[inst_ind, 1+3*activation_ind+alpha_ind] = str(val_accuracy)[0:4]
                            train_accuracy_list.append(train_accuracy); val_accuracy_list.append(val_accuracy)
    np.savetxt(filename, results_arr, delimiter=' & ', fmt = '%s', newline=' \\\\\n')
    num_models = len(train_accuracy_list)
    font = {'family' : 'serif', 'size'   : 16}

    plt.clf()
    plt.scatter(np.linspace(1,num_models, num_models, endpoint=True), np.array(train_accuracy_list), s = 60, alpha = 0.6, marker = "o", label = "train")
    plt.scatter(np.linspace(1,num_models, num_models, endpoint=True), np.array(val_accuracy_list), s = 60, alpha = 0.6, marker = "d", label = "validation")
    plt.xlabel("model trials",font); plt.ylabel("accuracy",font); plt.legend(prop={'size': 16, 'family' : 'serif'});
    plt.xticks(fontsize=14); plt.yticks(fontsize=16); plt.ylim([0.65, 1]); plt.savefig("train_val_plot", bbox_inches = "tight")
# print("Unnormalized, Basic Features")
# run_nn_parameter_search(False, feature_type = "basic", filename = "log_lin_val_table_basic.csv")
print("Normalized, Basic Features")
run_nn_parameter_search(True, feature_type = "basic", type ="reg", filename = "nn_val_table_norm_basic_reg.csv")
run_nn_parameter_search(True, feature_type = "basic", type ="class", filename = "nn_val_table_norm_basic_class.csv")
# print("Unnormalized, Technical Features")
# run_nn_parameter_search(False, feature_type = "technical", filename = "log_lin_val_table_technical.csv")
print("Normalized, Technical Features")
run_nn_parameter_search(True, feature_type = "technical", type ="reg", filename = "nn_val_table_norm_technical_reg.csv")
run_nn_parameter_search(True, feature_type = "technical", type ="class", filename = "nn_val_table_norm_technical_class.csv")
