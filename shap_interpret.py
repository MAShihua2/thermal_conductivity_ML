import sklearn
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import numpy as np
from utils import PCA_dim_reducer, split_data, prepare_data_for_torch_model, evaluate
from torch_model import Dual_MLP_T_Emb
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from utils import get_ule_repre
import seaborn as sns
import os
import shap
import interpret.glassbox
from sklearn import preprocessing as pre

def merge_sro(X):
    merge_list = [X[:, 0], # 11
                  X[:, 1] + X[:, 4], # 12
                  X[:, 2] + X[:, 8], # 13
                  X[:, 3] + X[:, 12], # 14
                  X[:, 5], # 22
                  X[:, 6] + X[:, 9], # 23
                  X[:, 7] + X[:, 13], # 24
                  X[:, 10], # 33
                  X[:, 11] + X[:, 14], # 34
                  X[:, 15], # 44
                  ]
    
    return np.concatenate([item.reshape(-1, 1) for item in merge_list], axis=1)

def main():
    types_num = 4
    local_env = "sro_5nn"
    target = "soap" # or local_env
    # target = local_env
    ele_map = {1:"Ta", 2:"Nb", 3:"Mo", 4:"W"}
    if local_env == "sro_5nn":
        local_env_dim = 80
        bond_type_pairs = [f"{ele_map[i+1]}-{ele_map[j+1]}" for i in range(types_num) for j in range(types_num)]
    elif local_env == "bond":
        local_env_dim = 10
        bond_type_pairs = [f"{ele_map[i+1]}-{ele_map[j+1]}" for i in range(types_num) for j in range(types_num) if i <= j]
    bond_type_pairs = [f"{ele_map[i+1]}-{ele_map[j+1]}" for i in range(types_num) for j in range(types_num) if i <= j]
    struct_list = ["b2", "random", "S300", "S500", "S900", "S1100"]
    
    data_dir = f'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240_SHAP/results/ule_features/{local_env}'
    soap_features = np.load(f"{data_dir}/soap.npy")
    local_env_features = np.load(f"{data_dir}/{local_env}.npy")
    temperatures = np.load(f"{data_dir}/temp.npy")
    structures = np.load(f"{data_dir}/struct.npy")
    labels = np.load(f"{data_dir}/label.npy")


    ule_soap_local_env_representation = get_ule_repre(local_env_type=local_env, target=target)
    ule_local_env_features = ule_soap_local_env_representation[:, :local_env_dim]
    # normal ule_local_env_features
    if local_env == "sro_5nn":
        ule_local_env_features *= -1 # the initial range is (-3, 1)， now （-1， 3）
        ule_local_env_features = (ule_local_env_features + 1) / 4 # （0， 4）
    # merge same bonds
    merge_ule_local_env_features = np.concatenate([merge_sro(ule_local_env_features[:, :16]),
                                    merge_sro(ule_local_env_features[:, 16:32]),
                                    merge_sro(ule_local_env_features[:, 32:48]),
                                    merge_sro(ule_local_env_features[:, 48:64]),
                                    merge_sro(ule_local_env_features[:, 64:])], axis=1)
    # print(sum(merge_ule_local_env_features[0]))


    save_dir = f'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240_SHAP/results/interpret/{target}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # print(local_env_features.shape)

    # sample_ind = 20
    model = interpret.glassbox.ExplainableBoostingRegressor(interactions=0)
    # X = local_env_features
    X = soap_features
    # print(np.linalg.norm(X, axis=1))
    # print(X.shape)
    # exit()
    y = labels
    model.fit(X, y)
    scores = model.score(X, y)
    print(scores)

    X_sample = shap.utils.sample(X, 200)
    explainer = shap.Explainer(model.predict, X_sample)
    # shap_values = explainer(X)

    # all
    f = open(f"{save_dir}/{target}.txt", "w")
    shap_values = explainer(X)
    values = shap_values.values

    data_sum = values.sum(axis=0).reshape(1, -1)
    f.write(f"{' '.join([str(item) for item in data_sum.reshape(-1).tolist()])}")
    f.close()
    
    plt.clf()
    plt.cla()
    plt.figure(figsize=(100, 5))
    sns.heatmap(data=data_sum,cmap="RdBu_r",
                xticklabels=list(range(100)))
    plt.savefig(f"{save_dir}/{target}_u_weight.jpg")
    plt.close()
    # exit()
    #####################################################  struct ###########################################################

    f = open(f"{save_dir}/struct/struct_bond.txt", "w")
    struct_idx_dict = dict()
    struct_ule_shap_values = dict()
    for idx, struct in enumerate(struct_list):
        struct_idx = structures == idx
        struct_idx_dict[struct]= struct_idx
        struct_X = X[struct_idx]

        # 1
        shap_values = explainer(struct_X)
        values = shap_values.values

        data_sum = values.sum(axis=0).reshape(1, -1)
        struct_ule_shap_values[struct] = data_sum # 

        # 2
        if "sro" not in target and "bond" not in target:
            continue
        plt.clf()
        plt.cla()
        plt.figure(figsize=(10, 5))
        data_sum = (values @ merge_ule_local_env_features).sum(axis=0).reshape(5, -1)

        f.write(f"{struct} {' '.join([str(item) for item in data_sum.reshape(-1).tolist()])}\n")
        # data_sum = pre.MinMaxScaler().fit_transform(data_sum)
        
        ax = sns.heatmap(data=data_sum,cmap="RdBu_r", 
                    xticklabels=bond_type_pairs, 
                    yticklabels=["1NN", "2NN", "3NN", "4NN", "5NN"], 
                    vmin=-2,
                    vmax=5)
        ax.set_xticklabels(bond_type_pairs)
        ax.set_yticklabels(["1NN", "2NN", "3NN", "4NN", "5NN"])
        # plt.legend()
        plt.savefig(f"{save_dir}/struct/{struct}_bond_weight.jpg")
        plt.close()

    plot_struct_list = ["b2", "S300", "S500", "S900", "S1100", "random"]

    f = open(f"{save_dir}/struct/struct_ule.txt", "w")
    data_sum = []
    for struct in plot_struct_list:
        data_sum.append(struct_ule_shap_values[struct])
        f.write(f"{struct} {' '.join([str(item) for item in struct_ule_shap_values[struct].reshape(-1).tolist()])}\n")
    data_sum = np.concatenate(data_sum, axis=0)
    # data_sum = pre.MinMaxScaler().fit_transform(data_sum)
    plt.clf()
    plt.cla()
    plt.figure(figsize=(100, 5))
    sns.heatmap(data=data_sum,cmap="RdBu_r",
                xticklabels=list(range(100)),
                yticklabels=plot_struct_list)
    plt.savefig(f"{save_dir}/struct/struct_ule_weight.jpg")
    plt.close()


    ##################################################### temperature ###########################################################

    plot_temperature_list = [77, 150, 300, 500, 700, 900, 1300]
    temperature_idx_dict = dict()
    temperature_ule_shap_values = dict()
    f = open(f"{save_dir}/temperature/temperature_bond.txt", "w")
    for idx, temperature in enumerate(plot_temperature_list):
        temperature_idx = temperatures == temperature
        temperature_idx_dict[temperature]= temperature_idx
        temperature_X = X[temperature_idx]

        # 1
        shap_values = explainer(temperature_X)
        values = shap_values.values

        data_sum = values.sum(axis=0).reshape(1, -1)
        temperature_ule_shap_values[temperature] = data_sum

        if "sro" not in target and "bond" not in target:
            continue
        # 2
        plt.clf()
        plt.cla()
        plt.figure(figsize=(10, 5))
        data_sum = (values @ merge_ule_local_env_features).sum(axis=0).reshape(5, -1)
        f.write(f"{temperature} {' '.join([str(item) for item in data_sum.reshape(-1).tolist()])}\n")
        # print(data_sum.max())
        # print(data_sum.min())
        # data_sum = pre.MinMaxScaler().fit_transform(data_sum)
        ax = sns.heatmap(data=data_sum,cmap="RdBu_r", 
                    xticklabels=bond_type_pairs, 
                    yticklabels=["1NN", "2NN", "3NN", "4NN", "5NN"], 
                    vmin=-0.5,
                    vmax=3.5)
        ax.set_xticklabels(bond_type_pairs)
        ax.set_yticklabels(["1NN", "2NN", "3NN", "4NN", "5NN"])
        # plt.legend()
        plt.savefig(f"{save_dir}/temperature/{temperature}_bond_weight.jpg")
        plt.close()

    
    data_sum = []
    f = open(f"{save_dir}/temperature/temperature_ule.txt", "w")
    for temperature in plot_temperature_list:
        data_sum.append(temperature_ule_shap_values[temperature])
        f.write(f"{temperature} {' '.join([str(item) for item in temperature_ule_shap_values[temperature].reshape(-1).tolist()])}\n")
    data_sum = np.concatenate(data_sum, axis=0)
    # data_sum = pre.MinMaxScaler().fit_transform(data_sum)
    plt.clf()
    plt.cla()
    plt.figure(figsize=(100, 5))
    sns.heatmap(data=data_sum,cmap="RdBu_r",
                xticklabels=list(range(100)),
                yticklabels=plot_temperature_list)
    plt.savefig(f"{save_dir}/temperature/temperature_ule_weight.jpg")
    plt.close()


        





main()