import os
import numpy as np
import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, KMeans
import argparse
import matplotlib.pyplot as plt
import scipy as sci
import json

test_sample_type_list = ["b2", "random", "S300", "S500", "S900", "S1100"]

def get_cluster_features(cluster_list, bond_distribution_list, soap_feature_list):

    cluster_ids = sorted(set(cluster_list), reverse=True)
    average_bond_distribution_list = []
    average_soap_feature_list = []
    for cluster_id in cluster_ids:
        cluster_indices = [i for i, cluster in enumerate(cluster_list) if cluster == cluster_id]
        bond_distribution = np.mean([bond_distribution_list[i] for i in cluster_indices], axis=0)
        soap_feature = np.mean([soap_feature_list[i] for i in cluster_indices], axis=0)
        average_bond_distribution_list.append(bond_distribution)
        average_soap_feature_list.append(soap_feature)

    return average_bond_distribution_list, average_soap_feature_list


def main(args):
    # prepare_data
    temptures = ["77K", "150K", "300K", "500K", "700K", "900K", "1300K"] # train
    saved_dir = r'D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results'
    used_features_type_list = ["soap","sro_5nn"]
    
    for used_features_type in used_features_type_list:
        print(used_features_type)
        if used_features_type == "soap":
            used_feature_dim = 360
        elif used_features_type == "sro_5nn":
            used_feature_dim = 80
        soap_feature_list = []
        local_env_distribution_list = []
        all_used_features = []
        cluster_num = 100
        atom_num = 5000 #if used_features_type == "sro_5nn" else 10240
        soap_dim = 360
        local_env_dim = 80 #if used_features_type == "sro_5nn" else 10 
        train_data_dir = f'D:/Projects/shihuama_ml4thc/soap_csro/features/train/soap_sro_5nn'
        cluster_file = os.path.join(saved_dir, f"cluster_{used_features_type}.npy")
        print(cluster_file)
        for temp in temptures:
            print(temp)
            feature_file = os.path.join(train_data_dir, f"5_5_{temp}_features.npy")
            info_file = os.path.join(train_data_dir, f"5_5_{temp}_info.json")

            input_features = np.load(feature_file)
            print(input_features.shape)
            info_dict = json.load(open(info_file, "r"))
            labels = np.array(info_dict["label"])

            soap_local_env_features = input_features
            soap_local_env_features = soap_local_env_features.reshape(len(labels), atom_num, -1)
            soap_features = soap_local_env_features[:, :, :soap_dim]
            normalized_soap_features = soap_features / np.linalg.norm(soap_features, axis=2)[:, :, None]
            local_env_features = soap_local_env_features[:, :, soap_dim:]
            if used_features_type == "bond":
                normalized_local_env_features = local_env_features / np.sum(local_env_features, axis=2)[:, :, None]
            elif used_features_type == "sro_5nn":
                normalized_local_env_features = local_env_features
            elif used_features_type == "soap":
                normalized_local_env_features = local_env_features / np.sum(local_env_features, axis=2)[:, :, None]
                
            soap_local_env_features = np.concatenate((normalized_soap_features, normalized_local_env_features), axis=2)
            print(soap_local_env_features.shape)

            if used_features_type == "soap":
                used_features = normalized_soap_features
            elif used_features_type == "sro_5nn" or used_features_type == "bond":
                used_features = normalized_local_env_features
            elif used_features_type == "soap_sro_5nn":
                used_features = soap_local_env_features

            all_used_features.append(used_features)
            soap_feature_list.append(soap_features)
            local_env_distribution_list.append(normalized_local_env_features)
            
        print("Finish Aggregate")
        all_used_features = np.concatenate(all_used_features, axis=0) # n_samples x n_atoms x 10
        all_soap_features = np.concatenate(soap_feature_list, axis=0)
        all_local_env_features = np.concatenate(local_env_distribution_list, axis=0)

        print(all_local_env_features.shape)
        
        all_atom_used_features = all_used_features.reshape(-1, used_feature_dim)
        all_atom_soap_features = all_soap_features.reshape(-1, soap_dim)
        all_atom_local_env_features = all_local_env_features.reshape(-1, local_env_dim)


        # print(all_atom_local_env_features[:10])
        
        # clustering_model = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None)

        if cluster_num > 0:
            clustering_model = KMeans(n_clusters=cluster_num)
            clustering_model.fit(all_atom_used_features)
            cluster_id = clustering_model.labels_
            cluster_number = len(set(cluster_id))
            print("cluster_num: ", cluster_number)

            average_cluster_features = get_cluster_features(cluster_id, all_atom_local_env_features, all_atom_soap_features)

        else:
            num_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250, 300]
            js_values = []
            average_cluster_features_list = []
            print(f"select cluster num from {num_list}")
            f = open(f"D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn/results/JSD_{used_features_type}.txt", "w")
            for item in num_list:
                clustering_model = KMeans(n_clusters=item)
                clustering_model.fit(all_atom_used_features)
                cluster_id = clustering_model.labels_

                average_cluster_features = get_cluster_features(cluster_id, all_atom_local_env_features, all_atom_soap_features)
                average_local_env_list = average_cluster_features[0]

                # compute JS divergence between different clusters's average bond distribution
                js_divergence_dict = {}
                distance_f = sci.spatial.distance.jensenshannon if used_features_type == "bond" else sci.spatial.distance.cosine
                for j in range(len(average_local_env_list)):
                    for k in range(j+1, len(average_local_env_list)):

                        js_divergence = distance_f(average_local_env_list[j], average_local_env_list[k])
                        js_divergence_dict[(j, k)] = js_divergence
                
                average = np.mean(list(js_divergence_dict.values()))
                # print and add green color to variable in the string
                print(f"Average JS Divergence is \033[1;31;40m{average} for cluster num {item}")
                # print(f"Average JS Divergence is {average} for Sample {test_sample_type_list[i]} in Temperature {temp} with Cluster Number {cluster_number} based on {used_features_type}.")
                f.write(f"{average} {item}\n")
                js_values.append(average)
                average_cluster_features_list.append(average_cluster_features)

            best_index = js_values.index(max(js_values))
            cluster_num = num_list[best_index]
            average_cluster_features = average_cluster_features_list[best_index]
            print("best cluster_num is : ", cluster_num)

        if not os.path.isdir(f"{saved_dir}/cluster_{used_features_type}_fig/"):
            os.makedirs(f"{saved_dir}/cluster_{used_features_type}_fig/")

        for i, local_env_d in enumerate(average_cluster_features[0]):
            plt.clf()
            plt.cla()
            plt.figure()
            plt.bar(range(local_env_dim), local_env_d)
            plt.savefig(f"{saved_dir}/cluster_{used_features_type}_fig/{i}.png")
            plt.close()
            
        
        np.save(cluster_file, np.concatenate(average_cluster_features, axis=1))



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOAP bond")
    parser.add_argument("--project_dir", type=str, default="/home/sunzhen/Project/heat")
    parser.add_argument("--feature_type", type=str, default="sro")
    parser.add_argument("--avg_type", type=str, default="mean")
    parser.add_argument("--rcut", type=float, default=6.0)
    parser.add_argument("--nmax", type=int, default=6)
    parser.add_argument("--lmax", type=int, default=6)
    parser.add_argument("--feature_npy_file", type=str, default="")
    parser.add_argument("--label_npy_file", type=str, default="")

    args = parser.parse_args()
    main(args)
