import json
import os
import numpy as np
from utils import get_all_train_features_labels, get_ule_repre

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


ule_bond_soap_rep = get_ule_repre("sro_5nn")
ule_bond_distribution_list =ule_bond_soap_rep[:, :80]
ule_soap_list = ule_bond_soap_rep[:, 80:]
print(ule_soap_list.shape)


ule_bond_distribution_list *= -1 # the initial range is (-3, 1)， now （-1， 3）
ule_bond_distribution_list = (ule_bond_distribution_list + 1) / 4 # （0， 4）
# merge same bonds
ule_bond_distribution_list = np.concatenate([merge_sro(ule_bond_distribution_list[:, :16]),
                                merge_sro(ule_bond_distribution_list[:, 16:32]),
                                merge_sro(ule_bond_distribution_list[:, 32:48]),
                                merge_sro(ule_bond_distribution_list[:, 48:64]),
                                merge_sro(ule_bond_distribution_list[:, 64:])], axis=1)

structures = ["b2", "random", "S300", "S500", "S900", "S1100"]
temptures = ["77K", "150K", "300K", "500K", "700K", "900K", "1300K"] # train
# temptures = ["77K", "150K"] # test
train_data_dir=r'D:/Projects/shihuama_ml4thc/soap_csro/features/train/soap_sro_5nn'
soap_feature_list = []
bond_distribution_list = []
label_list = []
temperature_list = []
structure_list = []

save_dir = r'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240_SHAP/results/ule_atoms'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

f = open(f"{save_dir}/soap_ule_id-file_index-atom_index.txt", "w")
topk = 5
for j, ule_bond in enumerate(ule_bond_distribution_list):
    ule_soap = ule_soap_list[j]
    data_file_index_list = []
    atom_index_list = []
    max_sim_list = []
    bond_feature_list = []
    for temp in temptures:
        feature_file = os.path.join(train_data_dir, f"5_5_{temp}_features.npy")
        info_file = os.path.join(train_data_dir, f"5_5_{temp}_info.json")

        input_features = np.load(feature_file)
        info_dict = json.load(open(info_file, "r"))
        labels = np.array(info_dict["label"])
        select_indexs = np.array(info_dict["select_index"])

        soap_bond_features = input_features

        bond_features = soap_bond_features[:, 360:]
        soap_features = soap_bond_features[:, :360]

        bond_features = np.concatenate([merge_sro(bond_features[:, :16]),
                                        merge_sro(bond_features[:, 16:32]),
                                        merge_sro(bond_features[:, 32:48]),
                                        merge_sro(bond_features[:, 48:64]),
                                        merge_sro(bond_features[:, 64:])], axis=1)

        bond_features = bond_features.reshape(len(labels), select_indexs.shape[1], -1)
        soap_features = soap_features.reshape(len(labels), select_indexs.shape[1], -1)
        # bond_features = soap_bond_features[:, :, 360:]

        bond_features *= -1 # the initial range is (-3, 1)， now （-1， 3）
        bond_features = (bond_features + 1) / 4 # （0， 4）
    # merge same bonds


        temperature = np.array(info_dict["temperature"])
        structure = info_dict["structure"]

        offset = 0
        last_struct = ""
        for i, struct in enumerate(structure):
            if struct != last_struct:
                offset = i
                last_struct = struct
            data_file_index = f"{temp}-{struct}-{i-offset+1}"
            bond_features_i = bond_features[i] # 10240x10
            soap_features_i = soap_features[i]
            # print(bond_features_i.shape)
            # print(ule_bond.shape)
            bond_similarity = (bond_features_i * ule_bond).sum(axis=-1)
            soap_similarity = (soap_features_i * ule_soap).sum(axis=-1)
            # print(similarity.shape)
            similarity = soap_similarity
            atom_index = np.argmax(similarity.reshape(-1))
            max_sim = similarity.reshape(-1)[atom_index]
            bond_feature = bond_features_i[atom_index]
            
            # print(f"Current struct is {temp}-{struct}-{i-offset+1}")

            data_file_index_list.append(data_file_index)
            atom_index_list.append(select_indexs[i][atom_index])
            max_sim_list.append(max_sim)
            bond_feature_list.append(bond_feature)

    data_file_index_array = np.array(data_file_index_list)
    atom_index_array = np.array(atom_index_list)
    max_sim_array = np.array(max_sim_list)
    # bond_feature_array = np.array(bond_feature_list)
    topk_index = np.argpartition(max_sim_array, -topk)[-topk:]
    print(max_sim_array[topk_index])

    for index in topk_index:
        f.write(f"{j} {data_file_index_array[index]} {atom_index_array[index]} {' '.join([str(item) for item in bond_feature_list[index]])}\n")
        f.flush()



