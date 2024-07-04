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


def interpret_by_thc_range(local_env, save_dir, ule_local_env_features, sum_features, labels, bond_type_pairs, ule_threshold=0.5):
    tc_range_list = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    tc_label_list = []
    ule_scores_list = []
    bond_scores_list = []
    for i in range(len(tc_range_list)-1):
        cur_range_index = labels > tc_range_list[i]
        cur_range_index = cur_range_index & (labels < tc_range_list[i+1])

        tc_label_list.append(f"{tc_range_list[i]}-{tc_range_list[i+1]}")

        cur_range_features = sum_features[cur_range_index]
        cur_range_labels = labels[cur_range_index]

        interpret_model = SelectKBest(f_regression, k='all')
        interpret_model.fit(cur_range_features, cur_range_labels)

        ule_scores = interpret_model.scores_
        ule_scores = ule_scores / ule_scores.max()
        # ule_scores = np.exp(ule_scores) / np.exp(ule_scores - ule_scores.max()).sum()

        ule_scores_list.append(ule_scores)
        most_important_ules_index = ule_scores > ule_threshold
        most_important_ule_bonds = ule_local_env_features[most_important_ules_index]
        bonds_scores = most_important_ule_bonds.sum(axis=0)
        bonds_scores /= bonds_scores.max() if local_env == "bond" else bonds_scores.sum()
        bond_scores_list.append(bonds_scores)

    plt.clf()
    plt.cla()
    data = np.array(ule_scores_list)
    sns.heatmap(data=data,cmap="RdBu_r", yticklabels=tc_label_list)
    plt.savefig(f"{save_dir}/ule_tc.jpg")

    plt.clf()
    plt.cla()
    plt.figure(figsize=(20, 1))
    data_sum = data.sum(axis=0).reshape(1, -1)
    data_sum /= data_sum.max()
    sns.heatmap(data=data_sum,cmap="RdBu_r")
    plt.savefig(f"{save_dir}/all_ule_tc.jpg")

    ule_tc_f = open(f"{save_dir}/ule_tc.txt", "w")
    for i, ule_scores in enumerate(ule_scores_list):
        ule_tc_f.write(f"{tc_label_list[i]} {' '.join([str(item) for item in ule_scores.tolist()])}\n")

    all_ule_tc_f = open(f"{save_dir}/all_ule_tc.txt", "w")
    all_ule_tc_f.write(f"{' '.join([str(item) for item in data_sum.reshape(-1).tolist()])}\n")

    

    plt.clf()
    plt.cla()
    data = np.array(bond_scores_list)
    sns.heatmap(data=data,cmap="RdBu_r", yticklabels=tc_label_list, xticklabels=bond_type_pairs)
    plt.savefig(f"{save_dir}/bond_tc.jpg")

    bond_tc_f = open(f"{save_dir}/bond_tc.txt", "w")
    for i, bond_scores in enumerate(bond_scores_list):
        bond_tc_f.write(f"{tc_label_list[i]} {' '.join([str(item) for item in bond_scores.tolist()])}\n")


def plot_and_write_heatmap(ule_scores_list, bond_scores_list, target_set, save_dir, prefix, bond_type_pairs):

    plt.clf()
    plt.cla()
    data = np.array(ule_scores_list)
    sns.heatmap(data=data,cmap="RdBu_r", yticklabels=target_set)
    plt.savefig(f"{save_dir}/{prefix}_ule_tc.jpg")

    ule_tc_f = open(f"{save_dir}/{prefix}_ule_tc.txt", "w")
    for i, ule_scores in enumerate(ule_scores_list):
        ule_tc_f.write(f"{target_set[i]} {' '.join([str(item) for item in ule_scores.tolist()])}\n")


    plt.clf()
    plt.cla()
    data = np.array(bond_scores_list)
    sns.heatmap(data=data,cmap="RdBu_r", yticklabels=target_set, xticklabels=bond_type_pairs)
    plt.savefig(f"{save_dir}/{prefix}_bond_tc.jpg")

    bond_tc_f = open(f"{save_dir}/{prefix}_bond_tc.txt", "w")
    for i, bond_scores in enumerate(bond_scores_list):
        bond_tc_f.write(f"{target_set[i]} {' '.join([str(item) for item in bond_scores.tolist()])}\n")


def interpret_by_type(local_env, target, input_features, ule_local_env_features, labels, ule_threshold, T):
    print(target)
    target_set = sorted(list(set(target)), reverse=True)
    ule_scores_list = []
    bond_scores_list = []
    for i, item in enumerate(target_set):
        cur_index = target == item
        cur_range_features = input_features[cur_index]
        cur_range_labels = labels[cur_index]

        interpret_model = SelectKBest(r_regression, k='all')
        interpret_model.fit(cur_range_features, cur_range_labels)

        # get ule contributions
        ule_scores = interpret_model.scores_
        # ule_scores = ule_scores / ule_scores.max()
        ule_scores = (ule_scores - ule_scores.min()) / (ule_scores.max() - ule_scores.min())
        uel_scores = np.exp(ule_scores) / np.exp(ule_scores).sum()
        # print(ule_scores)

        # get bond contributions
        most_important_ules_index = ule_scores > ule_threshold
        most_important_ule_bonds = ule_local_env_features[most_important_ules_index]
        bonds_scores = most_important_ule_bonds.sum(axis=0)
        bonds_scores /= bonds_scores.max()

        ule_scores_list.append(ule_scores)
        bond_scores_list.append(bonds_scores)

    return ule_scores_list, bond_scores_list


def main():
    types_num = 4
    local_env = "sro_5nn"
    ele_map = {1:"Ta", 2:"Nb", 3:"Mo", 4:"W"}
    if local_env == "sro_5nn":
        local_env_dim = 80
        bond_type_pairs = [f"{ele_map[i+1]}-{ele_map[j+1]}" for i in range(types_num) for j in range(types_num)]
    elif local_env == "bond":
        local_env_dim = 10
        bond_type_pairs = [f"{ele_map[i+1]}-{ele_map[j+1]}" for i in range(types_num) for j in range(types_num) if i <= j]
    
    struct_list = ["b2", "random", "S300", "S500", "S900", "S1100"]
    
    data_dir = f'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/ule_features/{local_env}'
    soap_features = np.load(f"{data_dir}/soap.npy")
    local_env_features = np.load(f"{data_dir}/{local_env}.npy")
    temperatures = np.load(f"{data_dir}/temp.npy")
    structures = np.load(f"{data_dir}/struct.npy")
    labels = np.load(f"{data_dir}/label.npy")

    
    # preprocess features
    # features = np.concatenate([soap_features, bond_features, temperatures], axis=-1)
    relation_model_dict = {
        "f_regression": f_regression,
        "mutual_info_regression": mutual_info_regression,
        "r_regression": r_regression

    }
    save_dir = f'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/interpret/{local_env}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # for k, v in relation_model_dict.items():
    #     interpret_model = SelectKBest(v, k='all')
    #     interpret_model.fit(bond_features+soap_features, labels)

    #     ule_scores = interpret_model.scores_
    #     ule_scores = ule_scores / ule_scores.max()
    #     plt.clf()
    #     plt.cla()
    #     plt.bar(range(100), ule_scores)
    #     plt.xlabel("ULE IDs")
    #     plt.ylabel("Importance")
    #     plt.savefig(f"{save_dir}/{k}.jpg")
    # print(local_env_features.shape)
    # print(soap_features.shape)
    

    ule_soap_local_env_representation = get_ule_repre(local_env_type=local_env)
    ule_local_env_features = ule_soap_local_env_representation[:, :local_env_dim]
    # normal ule_local_env_features
    if local_env == "sro_5nn":
        ule_local_env_features *= -1 # the initial range is (-3, 1)， now （-1， 3）
        ule_local_env_features = (ule_local_env_features + 1) / 4 # （0， 4）
    ule_soap_features = ule_soap_local_env_representation[:, local_env_dim:]

    sum_features = local_env_features+soap_features if local_env == "bond" else local_env_features

    ule_threshold = 0.5
    T = 1

    interpret_by_thc_range(local_env, save_dir, ule_local_env_features, sum_features, labels, bond_type_pairs, ule_threshold)

    # temperatures
    prefix = "temp"
    target = temperatures
    input_features = sum_features
    target_set = sorted(list(set(target)), reverse=True)
    ule_scores_list, bond_scores_list = interpret_by_type(local_env, target, input_features, ule_local_env_features, labels, ule_threshold, T)
    plot_and_write_heatmap(ule_scores_list, bond_scores_list, target_set, save_dir, prefix, bond_type_pairs)


    # structures
    prefix = "struct"
    target = structures
    input_features = sum_features
    target_set = [struct_list[item] for item in sorted(list(set(target)), reverse=True)]
    ule_scores_list, bond_scores_list = interpret_by_type(local_env, target, input_features, ule_local_env_features, labels, ule_threshold, T)
    plot_and_write_heatmap(ule_scores_list, bond_scores_list, target_set, save_dir, prefix, bond_type_pairs)

    # temp_structures
    prefix = "temp_struct"
    save_dir = f"{save_dir}/{prefix}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for temp in sorted(list(set(temperatures)), reverse=True):
        prefix = f"{temp}_struct"
        temp_index = temperatures == temp
        input_features = sum_features[temp_index]
        target = structures[temp_index]
        temp_labels = labels[temp_index]
        temp_ule_local_env_features = ule_local_env_features
        ule_scores_list, bond_scores_list = interpret_by_type(local_env, target, input_features, temp_ule_local_env_features, temp_labels, ule_threshold, T)
        plot_and_write_heatmap(ule_scores_list, bond_scores_list, target_set, save_dir, prefix, bond_type_pairs)

    



        





main()