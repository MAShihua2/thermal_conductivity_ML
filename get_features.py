import os
import time
import numpy as np
import json
from tqdm import tqdm
from utils import extract_soap_features
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lmax", type=int, default=5)
parser.add_argument("--nmax", type=int, default=5)
parser.add_argument("--feature_type", type=str, default="soap_sro_5nn")
parser.add_argument("--avg_type", type=str, default="inner")

args = parser.parse_args()

# for training data
rcut_dict = {
    "random": 6.788,
    "S1300": 6.84,
    "S1100": 6.84,
    "S900": 6.825,
    "S700": 6.81,
    "S500": 6.81,
    "S300": 6.80,
    "S77": 6.79,
    "b2": 6.689
}

project_dir=r'D:\Projects\shihuama_ml4thc\soap_csro'
nmax = args.nmax
lmax = args.lmax
avg_type = args.avg_type
feature_type = args.feature_type

s = time.time()
situation_type_list = ["77K", "150K", "300K", "500K", "700K", "900K", "1300K"]
test_sample_type_list = ["b2", "random", "S300", "S500", "S900", "S1100"]


for situation_type in situation_type_list:
    # loop situation type list

    # initialize storers and output files
    ## for features
    sro_feature_list = []
    soap_feature_list = []
    select_index_list = []
    output_features_file = os.path.join(project_dir, "features", "train/{}".format(feature_type), "{}_{}_{}_features.npy".format(nmax, lmax, situation_type))

    ## for other information
    structure_list = []
    temperature_list = []
    labels = []
    output_dict = dict()
    output_info_file = os.path.join(project_dir, "features", "train/{}".format(feature_type), "{}_{}_{}_info.json".format(nmax, lmax, situation_type))
    
    # check file directory
    target_dir = os.path.join(project_dir, "data", situation_type)
    assert os.path.exists(target_dir), "Target dir {} does not exist!".format(target_dir)

    for test_sample_type in tqdm(test_sample_type_list):
        # loop sample type list
        r_cut = rcut_dict[test_sample_type] # get r_cut for this sample_type

        print("Start {} in {}".format(test_sample_type, situation_type))

        # check file directory
        test_sample_dir = os.path.join(target_dir, test_sample_type)
        assert os.path.exists(test_sample_dir), "Test sample dir {} does not exist!".format(test_sample_dir)
        thermal_conductivity_file = os.path.join(test_sample_dir, "thermal_conductivity.txt")
        assert os.path.exists(thermal_conductivity_file), "Thermal conductivity file {} does not exist!".format(thermal_conductivity_file)

        # get sopa features for test samples
        # SXX and b2, whose data files are in model
        sample_model_data_file_path_template = ""
        if ("S" in test_sample_type or "b2" in test_sample_type) or situation_type in ["77K", "150K"]:

            model_data_dir = os.path.join(project_dir, "data/model")
            assert os.path.exists(model_data_dir), "Model data dir {} does not exist!".format(model_data_dir)

            sample_model_data_file_path_template = os.path.join(model_data_dir, 
                    "{}K_{}.data".format(test_sample_type.replace("S", ""), "system_id") if "S" in test_sample_type else
                    "{}_{}.data".format(test_sample_type, "system_id"))
        else:
            sample_model_data_file_path_template = os.path.join(test_sample_dir, "system_id", "random.data")

        for line in tqdm(open(thermal_conductivity_file, "r").readlines()):
            # loop tc (label) lines and get features for each line, each line is a system
            system_id, thermal_conductivity = line.strip().split()
            label = eval(thermal_conductivity)
            sys_file = sample_model_data_file_path_template.replace("system_id", str(system_id))

            assert os.path.exists(sys_file), "System file {} does not exist!".format(sys_file)
            # extract_sro_features(sys_file, lmax=lmax, nmax=nmax, rcut=r_cut, avg_type=avg_type)
            
            soap_features, pos_vector, sro_features, select_index = extract_soap_features(sys_file, lmax=lmax, nmax=nmax, rcut=r_cut, avg_type=avg_type)
            soap_feature_list.append(soap_features[select_index][:5000])
            sro_feature_list.append(sro_features[select_index][:5000])
            select_index_list.append(np.array(range(len(select_index)))[select_index][:5000].tolist())

            structure_list.append(test_sample_type)
            temperature_list.append(eval(situation_type.replace("K", "")))
            labels.append(label)

        print("Finish {} in {}".format(test_sample_type, situation_type))

    # prepare and save data
    soap_features_array = np.concatenate(soap_feature_list, axis=0)
    sro_features_array = np.concatenate(sro_feature_list, axis=0)

    print(soap_features_array.shape)
    print(sro_features_array.shape)

    soap_bond_features_array = np.concatenate([soap_features_array, sro_features_array], axis=-1)
    
    output_dict["structure"] = structure_list
    output_dict["temperature"] = temperature_list
    output_dict["label"] = labels
    output_dict["select_index"] = select_index_list

    target_dir = r'../features/train/{}'.format(feature_type)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    np.save(output_features_file, soap_bond_features_array)
    json.dump(output_dict, open(output_info_file, "w"), ensure_ascii=True, indent=4)

e = time.time()
print("Cost {0} mins to generate data".format((e-s)/60))


# for test data
exit()

situation_type_list = ["77K", "150K"]
for situation_type in situation_type_list:
    bond_feature_list = []
    soap_feature_list = []
    structure_list = []
    temperature_list = []
    labels = []

    output_dict = dict()
    output_features_file = os.path.join(project_dir, "features", "test/{}".format(feature_type), "{}_{}_{}_features.npy".format(nmax, lmax, situation_type))
    output_info_file = os.path.join(project_dir, "features", "test/{}".format(feature_type), "{}_{}_{}_info.json".format(nmax, lmax, situation_type))

    target_dir = os.path.join(r'/Users/shihuama/Downloads/ml4thc/size3', situation_type)
    assert os.path.exists(target_dir), "Target dir {} does not exist!".format(target_dir)

    thermal_conductivity_file = os.path.join(target_dir, "thermal_conductivity.txt")
    assert os.path.exists(thermal_conductivity_file), "Thermal conductivity file {} does not exist!".format(thermal_conductivity_file)

    for line in open(thermal_conductivity_file, "r").readlines():
        system_id, thermal_conductivity = line.strip().split()
        label = eval(thermal_conductivity)
        sys_file = os.path.join(target_dir, system_id, "model.data")

        assert os.path.exists(sys_file), "System file {} does not exist!".format(sys_file)
        soap_features, pos_vector, bond_features = extract_soap_features(sys_file, lmax=lmax, nmax=nmax, rcut=6.788, avg_type=avg_type)

        soap_feature_list.append(soap_features)
        bond_feature_list.append(bond_features)
        structure_list.append("test")
        temperature_list.append(eval(situation_type.replace("K", "")))
        labels.append(label)

    soap_features_array = np.concatenate(soap_feature_list, axis=0)
    bond_features_array = np.concatenate(bond_feature_list, axis=0)
    soap_bond_features_array = np.concatenate([soap_features_array, bond_features_array], axis=-1)

    output_dict["structure"] = structure_list
    output_dict["temperature"] = temperature_list
    output_dict["label"] = labels

    target_dir = r'../features/test/{}'.format(feature_type)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    np.save(output_features_file, soap_bond_features_array)
    json.dump(output_dict, open(output_info_file, "w"), ensure_ascii=True, indent=4)

