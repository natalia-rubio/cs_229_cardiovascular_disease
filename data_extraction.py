import numpy as np
from scipy.special import gamma
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
import pandas as pd
import random; random.seed(1)
from os.path import exists
font = {"size": 16}

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_inst_data(inst_name):
    attribute_list = [
        "id",
        "ccf",
        "age",
        "sex",
        "painloc",
        "painexer",
        "relrest",
        "pncaden",
        "cp",
        "trestbps",
        "htn",
        "chol",
        "smoke",
        "cigs",
        "years",
        "fbs",
        "dm",
        "famhist",
        "restecg",
        "ekgmo",
        "ekgday",
        "ekgyr",
        "dig",
        "prop",
        "nitr",
        "pro",
        "diuretic",
        "proto",
        "thaldur",
        "thaltime",
        "met",
        "thalach",
        "thalrest",
        "tpeakbps",
        "tpeakbpd",
        "dummy",
        "trestbpd",
        "exang",
        "xhypo",
        "oldpeak",
        "slope",
        "rldv5",
        "rldv5e",
        "ca",
        "restckm",
        "exerckm",
        "restef",
        "restwm",
        "exeref",
        "exerwm",
        "thal",
        "thalsev",
        "thalpul",
        "earlobe",
        "cmo",
        "cday",
        "cyr",
        "num",
        "lmt",
        "ladprox",
        "laddist",
        "diag",
        "cxmain",
        "ramus",
        "om1",
        "om2",
        "rcaprox",
        "rcadist",
        "lvx1",
        "lvx2",
        "lvx3",
        "lvx4",
        "lvf",
        "cathef",
        "junk"
    ]
    if inst_name == "all":
        raw_data = open('data/'+"cleveland"+'.data', "r").read().split('name') + \
        open('data/'+"hungarian"+'.data', "r").read().split('name') + \
        open('data/'+"long_beach"+'.data', "r").read().split('name') + \
        open('data/'+"switzerland"+'.data', "r").read().split('name')

    else:
        raw_data = open('data/'+inst_name+'.data', "r").read().split('name')

    num_samples = len(raw_data); num_attributes = len(attribute_list)
    print(inst_name + f" {num_samples} samples.")
    sample_arr = np.zeros((1,num_attributes))
    for sample in raw_data:
        sample.replace("\n", " ")
        attributes = np.float_(sample.split())
        if attributes.size != 75:
            print("missing data"); continue
        sample_arr = np.vstack((sample_arr, attributes.reshape(1, num_attributes)))
    sample_arr = sample_arr[1:,:]
    all_data = pd.DataFrame(sample_arr, columns = attribute_list)

    all_data["presence"] = list((np.asarray(all_data["num"])>0).astype(int))
    all_data["number"] = list((np.asarray(all_data["lmt"])>1).astype(int)
                            + (np.asarray(all_data["ladprox"])>1).astype(int)
                            + (np.asarray(all_data["laddist"])>1).astype(int)
                            + (np.asarray(all_data["cxmain"])>1).astype(int)
                            + (np.asarray(all_data["om1"])>1).astype(int)
                            + (np.asarray(all_data["rcaprox"])>1).astype(int)
                            + (np.asarray(all_data["rcadist"])>1).astype(int)
                            + (np.asarray(all_data["lvx1"])>1).astype(int)
                            + (np.asarray(all_data["lvx2"])>1).astype(int)
                            + (np.asarray(all_data["lvx3"])>1).astype(int)
                            + (np.asarray(all_data["lvx4"])>1).astype(int)
                            + (np.asarray(all_data["lvf"])>1).astype(int))

    indices = [i for i in range(len(all_data["number"]))]
    train_indices = random.sample(indices, int(num_samples*0.7));
    val_indices = list(set(indices) - set(train_indices))
    np.save("data/" + inst_name + "_train_indices", np.asarray(train_indices), allow_pickle = True)
    np.save("data/" + inst_name + "_val_indices", np.asarray(val_indices), allow_pickle = True)

    all_data.to_csv("data/" + inst_name + '_all_data.csv', header=True, index=False)

    basic_features = np.hstack((
                        all_data["age"].to_numpy().reshape(-1,1),
                        all_data["sex"].to_numpy().reshape(-1,1),
                        # all_data["painloc"].to_numpy().reshape(-1,1),
                        # all_data["painexer"].to_numpy().reshape(-1,1),
                        # all_data["relrest"].to_numpy().reshape(-1,1),
                        all_data["cp"].to_numpy().reshape(-1,1),
                        all_data["relrest"].to_numpy().reshape(-1,1),
                        all_data["cigs"].to_numpy().reshape(-1,1),
                        all_data["years"].to_numpy().reshape(-1,1),
                        all_data["famhist"].to_numpy().reshape(-1,1)));
    np.save("data/" + inst_name + "_basic_features", basic_features, allow_pickle = True)

    tech_all_features = np.hstack((
                        all_data["trestbps"].to_numpy().reshape(-1,1),
                        all_data["htn"].to_numpy().reshape(-1,1),
                        all_data["chol"].to_numpy().reshape(-1,1),
                        all_data["fbs"].to_numpy().reshape(-1,1),
                        all_data["restecg"].to_numpy().reshape(-1,1),
                        all_data["cp"].to_numpy().reshape(-1,1),
                        all_data["prop"].to_numpy().reshape(-1,1),
                        all_data["nitr"].to_numpy().reshape(-1,1),
                        all_data["pro"].to_numpy().reshape(-1,1),
                        all_data["diuretic"].to_numpy().reshape(-1,1),
                        all_data["proto"].to_numpy().reshape(-1,1),
                        all_data["thaldur"].to_numpy().reshape(-1,1),
                        all_data["thaltime"].to_numpy().reshape(-1,1),
                        all_data["met"].to_numpy().reshape(-1,1),
                        all_data["thalach"].to_numpy().reshape(-1,1),
                        all_data["tpeakbps"].to_numpy().reshape(-1,1),
                        all_data["tpeakbpd"].to_numpy().reshape(-1,1),
                        all_data["trestbpd"].to_numpy().reshape(-1,1),
                        all_data["exang"].to_numpy().reshape(-1,1),
                        all_data["xhypo"].to_numpy().reshape(-1,1),
                        all_data["oldpeak"].to_numpy().reshape(-1,1),
                        all_data["slope"].to_numpy().reshape(-1,1),
                        all_data["rldv5"].to_numpy().reshape(-1,1),
                        all_data["rldv5e"].to_numpy().reshape(-1,1),
                        all_data["thal"].to_numpy().reshape(-1,1)))
    np.save("data/" + inst_name + "_tech_all_features", tech_all_features, allow_pickle = True)

    tech_features = np.hstack((
                        all_data["age"].to_numpy().reshape(-1,1),
                        all_data["sex"].to_numpy().reshape(-1,1),
                        # all_data["painloc"].to_numpy().reshape(-1,1),
                        # all_data["painexer"].to_numpy().reshape(-1,1),
                        # all_data["relrest"].to_numpy().reshape(-1,1),
                        all_data["cp"].to_numpy().reshape(-1,1),
                        all_data["relrest"].to_numpy().reshape(-1,1),
                        all_data["cigs"].to_numpy().reshape(-1,1),
                        all_data["years"].to_numpy().reshape(-1,1),
                        all_data["famhist"].to_numpy().reshape(-1,1),
                        all_data["trestbps"].to_numpy().reshape(-1,1),
                        all_data["htn"].to_numpy().reshape(-1,1),
                        all_data["chol"].to_numpy().reshape(-1,1),
                        all_data["fbs"].to_numpy().reshape(-1,1),
                        all_data["restecg"].to_numpy().reshape(-1,1),
                        all_data["cp"].to_numpy().reshape(-1,1),
                        all_data["thaldur"].to_numpy().reshape(-1,1),
                        all_data["thaltime"].to_numpy().reshape(-1,1),
                        all_data["met"].to_numpy().reshape(-1,1),
                        all_data["thalach"].to_numpy().reshape(-1,1),
                        all_data["tpeakbps"].to_numpy().reshape(-1,1),
                        all_data["tpeakbpd"].to_numpy().reshape(-1,1),
                        all_data["trestbpd"].to_numpy().reshape(-1,1),
                        all_data["exang"].to_numpy().reshape(-1,1),
                        all_data["xhypo"].to_numpy().reshape(-1,1),
                        all_data["oldpeak"].to_numpy().reshape(-1,1),
                        all_data["slope"].to_numpy().reshape(-1,1),
                        all_data["rldv5"].to_numpy().reshape(-1,1),
                        all_data["rldv5e"].to_numpy().reshape(-1,1),
                        all_data["thal"].to_numpy().reshape(-1,1)))
    np.save("data/" + inst_name + "_tech_features", tech_features, allow_pickle = True)

    target_bin = np.asarray(all_data["presence"])
    np.save("data/" + inst_name + "_target_bin", target_bin, allow_pickle = True)
    target_cont = np.asarray(all_data["number"])
    np.save("data/" + inst_name + "_target_cont", target_cont, allow_pickle = True)
    return

def load_data(inst_name):
    X_basic = np.load("data/" + inst_name + "_basic_features.npy")
    X_technical = np.load("data/" + inst_name + "_tech_features.npy")
    target_bin = np.load("data/" + inst_name + "_target_bin.npy")
    target_cont = np.load("data/" + inst_name + "_target_cont.npy")
    train_indices = np.load("data/" + inst_name + "_train_indices.npy")
    val_indices = np.load("data/" + inst_name + "_val_indices.npy")
    return X_basic, X_technical, target_bin, target_cont, train_indices, val_indices

def main():
    if not exists("data/cleveland_all_data1.csv"):
        get_inst_data("cleveland")
    if not exists("data/hungarian_all_data1.csv"):
        get_inst_data("hungarian")
    if not exists("data/long_beach_all_data1.csv"):
        get_inst_data("long_beach")
    if not exists("data/switzerland_all_data1.csv"):
        get_inst_data("switzerland")
    if not exists("data/all_all_data1.csv"):
        get_inst_data("all")
    # *** END CODE HERE ***

if __name__ == '__main__':
    main()
