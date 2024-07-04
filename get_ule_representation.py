import os
import json
import numpy as np
from utils import get_all_train_features_labels, get_ule_repre


def get_representation(ule_features, structure_features):
    """
    ule_features: cluster_num x dim
    structure_features: N x atom_num x dim
    same dimension 
    """
    print(ule_features.shape)
    print(structure_features.shape)
    cluster_num, dim = ule_features.shape
    similarity = np.matmul(structure_features, ule_features.transpose(0, 1).reshape(1, dim, cluster_num))
    representation = similarity.sum(axis=1) # N x cluster_num
    return representation



def main():
    local_env = "sro_5nn"
    if local_env == "sro_5nn":
        local_env_dim = 80
    elif local_env == "bond":
        local_env_dim = 10
    else:
        print("The local environment is not sro and bond")
        exit()
    # get ule representation
    ule_soap_local_env_representation = get_ule_repre("soap")
    ule_soap_features = ule_soap_local_env_representation[:, local_env_dim:]

    ule_soap_local_env_representation = get_ule_repre(local_env)
    ule_local_env_features = ule_soap_local_env_representation[:, :local_env_dim]
    # get raw features, labels
    soap_features, local_env_features, labels, temperatures, structures = get_all_train_features_labels(local_env=local_env)

    # get ule representation
    soap_based_rep = get_representation(ule_soap_features, soap_features)
    local_env_based_rep = get_representation(ule_local_env_features, local_env_features)

    print(soap_based_rep.shape)
    print(local_env_based_rep.shape)

    save_dir = f"D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/ule_features/{local_env}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    soap_file = f"{save_dir}/soap.npy"
    local_env_file = f"{save_dir}/{local_env}.npy"
    label_file = f"{save_dir}/label.npy"
    temperature_file = f"{save_dir}/temp.npy"
    structure_file = f"{save_dir}/struct.npy"

    np.save(soap_file, soap_based_rep)
    np.save(local_env_file, local_env_based_rep)
    np.save(label_file, labels)
    np.save(temperature_file, temperatures)
    np.save(structure_file, structures)


main()
