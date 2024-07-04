import os
import json
import numpy as np
from ase.io import read, write
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
import torch
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matminer.featurizers.site.chemical import ChemicalSRO
from matminer.featurizers.conversions import ASEAtomstoStructure
from pymatgen.core import Lattice, Structure
import pandas as pd
import time
from pymatgen.analysis.local_env import CrystalNN
from collections import Counter


# aa2s = ASEAtomstoStructure()

def get_sro_distribution(bond_distribution, c_dict):
    n11 = bond_distribution[0]
    n12 = bond_distribution[1]
    n13 = bond_distribution[2]
    n14 = bond_distribution[3]
    n21 = bond_distribution[4]
    n22 = bond_distribution[5]
    n23 = bond_distribution[6]
    n24 = bond_distribution[7]
    n31 = bond_distribution[8]
    n32 = bond_distribution[9]
    n33 = bond_distribution[10]
    n34 = bond_distribution[11]
    n41 = bond_distribution[12]
    n42 = bond_distribution[13]
    n43 = bond_distribution[14]
    n44 = bond_distribution[15]

    ntotal = sum(bond_distribution)
    # if ntotal == 0 or 0 in list(c_dict.values()):
    #     print(ntotal)
    #     print(c_dict)
    if ntotal == 0:
        return [1.0] * 16

    sro11 = 1 - n11 / (ntotal * 1 * c_dict[1])
    sro12 = 1 - n12 / (ntotal * 1 * c_dict[2])
    sro13 = 1 - n13 / (ntotal * 1 * c_dict[3])
    sro14 = 1 - n14 / (ntotal * 1 * c_dict[4])
    sro21 = 1 - n21 / (ntotal * 1 * c_dict[1])
    sro22 = 1 - n22 / (ntotal * 1 * c_dict[2])
    sro23 = 1 - n23 / (ntotal * 1 * c_dict[3])
    sro24 = 1 - n24 / (ntotal * 1 * c_dict[4])
    sro31 = 1 - n31 / (ntotal * 1 * c_dict[1])
    sro32 = 1 - n32 / (ntotal * 1 * c_dict[2])
    sro33 = 1 - n33 / (ntotal * 1 * c_dict[3])
    sro34 = 1 - n34 / (ntotal * 1 * c_dict[4])
    sro41 = 1 - n41 / (ntotal * 1 * c_dict[1])
    sro42 = 1 - n42 / (ntotal * 1 * c_dict[2])
    sro43 = 1 - n43 / (ntotal * 1 * c_dict[3])
    sro44 = 1 - n44 / (ntotal * 1 * c_dict[4])
    
    return [sro11, sro12, sro13, sro14, sro21, sro22, sro23, sro24, sro31, sro32, sro33, sro34, sro41, sro42, sro43, sro44]


ele_dict = {1: "Ta",
            2: "Nb",
            3: "Mo",
            4: "W"}

def get_1nn_bond_features(positions, a_types, cut_off_1=3, cut_off_2=3.92, cut_off_3=5.05,cut_off_4=5.5,cut_off_5=6.15):
    sro_1_features = []
    sro_2_features = []
    sro_3_features = []
    sro_4_features = []
    sro_5_features = []
    types_num = len(set(a_types))
    bond_type_pairs = [(i+1, j+1) for i in range(types_num) for j in range(types_num)]
    c_dict = dict(Counter(a_types))
    for k, v in c_dict.items():
        c_dict[k] = v / len(a_types)
    num_1_nn = []
    num_2_nn = []
    num_3_nn = []
    num_4_nn = []
    num_5_nn = []
    for i in range(len(positions)):
        i_type = a_types[i]
        distance = np.linalg.norm(positions - positions[i], axis=1)

        nn_1_cut_index = np.where((distance < cut_off_1) & (distance > 0))[0]
        neigbor_features = np.zeros(len(bond_type_pairs))
        for j in nn_1_cut_index:
            j_type = a_types[j]  
            # key = (min(j_type, i_type), max(j_type, i_type))
            key = (i_type, j_type)
            neigbor_features[bond_type_pairs.index(key)] += 1
        num_1_nn.append(sum(neigbor_features))
        sro_features = get_sro_distribution(neigbor_features, c_dict)
        # print(sro_features)
        sro_1_features.append(sro_features)

        nn_2_cut_index = np.where((distance < cut_off_2) & (distance > cut_off_1))[0]
        neigbor_features = np.zeros(len(bond_type_pairs))
        for j in nn_2_cut_index:
            j_type = a_types[j]  
            # key = (min(j_type, i_type), max(j_type, i_type))
            key = (i_type, j_type)
            neigbor_features[bond_type_pairs.index(key)] += 1
        num_2_nn.append(sum(neigbor_features))
        sro_features = get_sro_distribution(neigbor_features, c_dict)
        # print(sro_features)
        sro_2_features.append(sro_features)
        # exit()

        nn_3_cut_index = np.where((distance < cut_off_3) & (distance > cut_off_2))[0]
        neigbor_features = np.zeros(len(bond_type_pairs))
        for j in nn_3_cut_index:
            j_type = a_types[j]  
            # key = (min(j_type, i_type), max(j_type, i_type))
            key = (i_type, j_type)
            neigbor_features[bond_type_pairs.index(key)] += 1
        num_3_nn.append(sum(neigbor_features))
        sro_features = get_sro_distribution(neigbor_features, c_dict)
        # print(sro_features)
        sro_3_features.append(sro_features)

        nn_4_cut_index = np.where((distance < cut_off_4) & (distance > cut_off_3))[0]
        neigbor_features = np.zeros(len(bond_type_pairs))
        for j in nn_4_cut_index:
            j_type = a_types[j]  
            # key = (min(j_type, i_type), max(j_type, i_type))
            key = (i_type, j_type)
            neigbor_features[bond_type_pairs.index(key)] += 1
        num_4_nn.append(sum(neigbor_features))
        sro_features = get_sro_distribution(neigbor_features, c_dict)
        # print(sro_features)
        sro_4_features.append(sro_features)

        nn_5_cut_index = np.where((distance < cut_off_5) & (distance > cut_off_4))[0]
        neigbor_features = np.zeros(len(bond_type_pairs))
        for j in nn_5_cut_index:
            j_type = a_types[j]  
            # key = (min(j_type, i_type), max(j_type, i_type))
            key = (i_type, j_type)
            neigbor_features[bond_type_pairs.index(key)] += 1
        num_5_nn.append(sum(neigbor_features))
        sro_features = get_sro_distribution(neigbor_features, c_dict)
        # print(sro_features)
        sro_5_features.append(sro_features)
        # exit()
    num_1nn_dict = dict(Counter(num_1_nn))
    num_2nn_dict = dict(Counter(num_2_nn))
    num_3nn_dict = dict(Counter(num_3_nn))
    num_4nn_dict = dict(Counter(num_4_nn))
    num_5nn_dict = dict(Counter(num_5_nn))
    # if 0 in num_1nn_dict:
    #     print(f"1nn num dict {num_1nn_dict}")
    # if 0 in num_2nn_dict:
    #     print(f"2nn num dict {num_2nn_dict}")
    num_2_nn_array = np.array(num_2_nn)
    num_2_nn_array_6_index = num_2_nn_array >=0 # 6
    # assert sum(num_2_nn_array_6_index) > 5000, print("Suitable atoms are too few")

    num_3_nn_array = np.array(num_3_nn)
    num_4_nn_array = np.array(num_4_nn)
    num_5_nn_array = np.array(num_5_nn)
      
    sro_1_features = np.array(sro_1_features)
    sro_2_features = np.array(sro_2_features)
    sro_3_features = np.array(sro_3_features)
    sro_4_features = np.array(sro_4_features)
    sro_5_features = np.array(sro_5_features)
    sro_features = np.concatenate([sro_1_features, sro_2_features, sro_3_features, sro_4_features, sro_5_features], axis=1)
    # index_range = np.array(range(len(num_2_nn_array_6_index)))[num_2_nn_array_6_index]

    # sro_features = sro_features[num_2_nn_array_6_index]
    # sro_features = sro_features[:5000]
    # select_index = index_range[:5000]
    # print(sro_features.shape)
    return sro_features, num_2_nn_array_6_index


def plot_points_line(x, y, x_label, y_label, title, save_path):
    plt.clf()
    plt.subplots(figsize=(10, 10))
    plt.scatter(x, y, s=10)
    plt.plot([0.5, 2.5], [0.5, 2.5], color="red")
    plt.xlim(1, 2.5)
    plt.ylim(1, 2.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def get_bond_features(positions, a_types, rcut):
    bond_features = []
    types_num = len(set(a_types))
    # bond_type_pairs = [(i, j) for i in range(types_num) for j in range(types_num)]
    bond_type_pairs = [(i+1, j+1) for i in range(types_num) for j in range(types_num) if i <= j]
    for i in range(len(positions)):
        distance = np.linalg.norm(positions - positions[i], axis=1)
        index = np.where(distance < rcut)[0]
        i_rcut_atoms = positions[index]
        neigbor_features = np.zeros(len(bond_type_pairs))
        for j in index:
            j_i_rcut_atoms_distance = np.linalg.norm(i_rcut_atoms - positions[j], axis=1)
            j_1_nn_index = np.where(j_i_rcut_atoms_distance < 3.05)[0]
            j_type = a_types[j]  
            neighbor_types = a_types[index[j_1_nn_index]]
            for item in neighbor_types:
                key = (min(j_type, item), max(j_type, item))
                neigbor_features[bond_type_pairs.index(key)] += 1
        bond_features.append(neigbor_features)
    
    bond_features = np.array(bond_features)
    return bond_features

def extract_soap_features(data_file, rcut, nmax, lmax, avg_type="inner"):
    test_file = data_file
    ase_sys = read(test_file, index=':5', format="lammps-data", style="atomic")
    
    positions = ase_sys[0].get_positions()
    atom_types = ase_sys[0].get_atomic_numbers()

    # bond_features = get_bond_features(positions, atom_types, rcut)
    sro_features, select_index = get_1nn_bond_features(positions, atom_types)

    soap_desc = SOAP(species=[1, 2, 3, 4], r_cut=rcut, n_max=nmax, l_max=lmax, compression={"mode":"crossover"})

    # Create descriptors as numpy arrays or sparse arrays
    soap = soap_desc.create(ase_sys)

    return soap, positions, sro_features, select_index


def get_all_train_features_labels(local_env="bond", norm_soap=False, norm_local_env=False):
    structures = ["b2", "random", "S300", "S500", "S900", "S1100"]
    temptures = ["77K", "150K", "300K", "500K", "700K", "900K", "1300K"] # train
    # temptures = ["77K", "150K"] # test

    if local_env == "bond":
        local_env_dim = 10
    elif local_env == "sro":
        local_env_dim = 32
    elif local_env == "sro_5nn":
        local_env_dim = 80
    soap_dim = 360
    train_data_dir = f'D:/Projects/shihuama_ml4thc/soap_csro/features/train/soap_{local_env}'
    soap_feature_list = []
    local_env_list = []
    label_list = []
    temperature_list = []
    structure_list = []
    for temp in temptures:
        feature_file = os.path.join(train_data_dir, f"5_5_{temp}_features.npy")
        info_file = os.path.join(train_data_dir, f"5_5_{temp}_info.json")

        input_features = np.load(feature_file)
        info_dict = json.load(open(info_file, "r"))
        labels = np.array(info_dict["label"])
        temperature = np.array(info_dict["temperature"])
        structure = np.array([structures.index(item) for item in info_dict["structure"]])

        soap_local_env_features = input_features
        soap_local_env_features = soap_local_env_features.reshape(len(labels), -1, soap_dim + local_env_dim)
        soap_features = soap_local_env_features[:, :, :soap_dim]
        local_env_features = soap_local_env_features[:, :, soap_dim:]

        if norm_local_env:
            local_env_features = local_env_features / np.sum(local_env_features, axis=2)[:, :, None]
        if norm_soap:
            soap_features = soap_features / np.linalg.norm(soap_features, axis=2)[:, :, None]

        soap_feature_list.append(soap_features)
        local_env_list.append(local_env_features)
        label_list.append(labels)
        temperature_list.append(temperature)
        structure_list.append(structure)

    all_soap_features = np.concatenate(soap_feature_list, axis=0)
    all_local_env_features = np.concatenate(local_env_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)
    all_temperatures = np.concatenate(temperature_list, axis=0)
    all_structures = np.concatenate(structure_list, axis=0)

    return all_soap_features, all_local_env_features, all_labels, all_temperatures, all_structures


def PCA_dim_reducer(features, dim):
    model = PCA(dim)
    redu_features = model.fit_transform(features)
    return redu_features

    
def split_data(features, ratio):
    train_len = int(len(features) * ratio[0])
    valid_len = int(len(features) * ratio[1])
    test_len = len(features) - train_len - valid_len

    indices = np.random.permutation(features.shape[0])
    return indices, train_len, valid_len, test_len

def prepare_data_for_torch_model(features, labels, index, train_len, valid_len):
    # shuffle data
    shuffle_features = features[index]
    shuffle_labels = labels[index]

    train_features = torch.from_numpy(shuffle_features[:train_len])
    train_labels = torch.from_numpy(shuffle_labels[:train_len])

    valid_features = torch.from_numpy(shuffle_features[train_len:train_len+valid_len])
    valid_labels = torch.from_numpy(shuffle_labels[train_len:train_len+valid_len])

    test_features = torch.from_numpy(shuffle_features[train_len+valid_len:])
    test_labels = torch.from_numpy(shuffle_labels[train_len+valid_len:])

    # convert numpy data to torch data
    train_dataset = TensorDataset(train_features.float(), train_labels.float())
    val_dataset = TensorDataset(valid_features.float(), valid_labels.float())
    test_dataset = TensorDataset(test_features.float(), test_labels.float())
    
    return train_dataset, val_dataset, test_dataset


def cal_regression_metrics(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    # r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return mae, mse, r2


def evaluate(model, data_loader, mode, device, criterion, save_dir, plot=False, log=False, epoch=None):
    model.eval()
    loss = 0
    outputs = []
    ys = []
    with torch.no_grad():
        for (X, y) in data_loader:
            out = model(X.to(device))
            loss += criterion(out, y.view(-1, 1).to(device)).item()
            outputs.append(out.detach().cpu().numpy())
            ys.append(y.numpy())
    outputs = np.concatenate(outputs, axis=0)
    ys = np.concatenate(ys, axis=0)

    mae, mse, r2 = cal_regression_metrics(ys, outputs)
    loss = loss / len(data_loader)

    if plot:
        plot_points_line(ys.reshape(-1), outputs.reshape(-1), "True", "Predict", "Predicted vs True (MAE: {:.4f}, r2: {:.3f})".format(mae, r2), f"{save_dir}/{mode}.png")

    if log:
        f = open("{}/{}_true_vs_predicted.txt".format(save_dir, mode), "w")
        f.write("True\tPredicted\n")
        for i in range(len(ys)):
            f.write("{}\t{}\n".format(ys[i], outputs[i][0]))
        f.close()

    # print("{} MAE: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(mode, mae, mse, r2))

    return mae, mse, r2, loss   



def get_ule_repre(local_env_type, target=None):
    if target is None:
        path = f'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/cluster_{local_env_type}.npy'
    else:
        path = f'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/cluster_{target}.npy'

    data = np.load(path)
    return data
