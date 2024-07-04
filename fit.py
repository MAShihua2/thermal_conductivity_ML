import sklearn
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import numpy as np
from utils import PCA_dim_reducer, split_data, prepare_data_for_torch_model, evaluate, cal_regression_metrics
from torch_model import Dual_MLP_T_Emb
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def log_plot(ys, outputs, mode, save_dir, mae, r2):
    plot_points_line(ys.reshape(-1), outputs.reshape(-1), "True", "Predict", "Predicted vs True (MAE: {:.4f}, r2: {:.3f})".format(mae, r2), f"{save_dir}/{mode}.png")

    f = open("{}/{}_true_vs_predicted.txt".format(save_dir, mode), "w")
    f.write("True\tPredicted\n")
    for i in range(len(ys)):
        f.write("{}\t{}\n".format(ys[i], outputs[i][0]))
    f.close()

def main():
    local_env = "sro_5nn"
    data_dir = f"D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/ule_features/{local_env}"
    soap_features = np.load(f"{data_dir}/soap.npy")
    local_env_features = np.load(f"{data_dir}/{local_env}.npy")
    temperatures = np.load(f"{data_dir}/temp.npy").reshape(-1, 1)
    labels = np.load(f"{data_dir}/label.npy")
    # k fold
    k =10
    # model paramters
    input_dim = 100
    hidden_dim = 256
    output_dim = 1
    drop = 0.15
    lr = 10e-5
    epoch_num = 5000
    temp_dim = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_type = "MHA_SOAP_CSRO_T_Emb"

    # preprocess features
    features = np.concatenate([soap_features, local_env_features, temperatures], axis=-1).astype(np.float32)
    # features = np.concatenate([local_env_features, local_env_features, temperatures], axis=-1) # just for bond
    # index, train_len, valid_len, test_len = split_data(features, [0.8, 0.1, 0.1])

    ##################################### test split for cross valid #####################
    split_index = int(len(features)* 1)
    fix_train_features = features[:split_index]
    fix_train_labels = labels[:split_index]
    fix_test_features = torch.from_numpy(features[split_index:])
    fix_test_labels = torch.from_numpy(labels[split_index:])

    fix_test_dataset = TensorDataset(fix_test_features, fix_test_labels)
    # shuffle data and convert numpy data to torch data
    # index, train_len, valid_len, test_len = split_data(fix_test_features, [0, 0, 0.1])
    ######################################################################################
    # train_dataset, val_dataset, test_dataset = prepare_data_for_torch_model(fix_test_dataset, fix_test_labels, index, train_len, valid_len)

    ##################################### test split for cross valid #####################
    # assert len(train_dataset) == 0 and len(val_dataset) == 0, print("Error split in fix test dataset")
    features = fix_train_features
    labels = fix_train_labels
    ######################################################################################

    fix_test_loader = DataLoader(fix_test_dataset, batch_size=32)

    print("*******")
    def prepare_k_fold(features, labels, k=10):
        data_loader_list = []
        print(features.shape)
        for i in range(k):
            # preprocess features
            index, train_len, valid_len, test_len = split_data(features, [0.9, 0, 0.1])
            # shuffle data and convert numpy data to torch data
            train_dataset, valid_dataset, test_dataset = prepare_data_for_torch_model(features, labels, index, train_len, valid_len)
            print("{} train data samples".format(len(train_dataset)))
            print("{} valid data samples".format(len(valid_dataset)))
            print("{} test data samples".format(len(test_dataset)))

            train_loader = DataLoader(train_dataset, batch_size=128)
            valid_loader = DataLoader(valid_dataset, batch_size=32)
            test_loader = DataLoader(test_dataset, batch_size=32)
            data_loader_list.append([train_loader, valid_loader, test_loader])
        return data_loader_list

    data_loader_list = prepare_k_fold(features, labels, k=k)
    save_dir = f"D:/Projects/shihuama_ml4thc/soap_csro/main_us_ucsro_5nn_10240/results/fit/soap_{local_env}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    k_valid_r2_list = [0] * k
    k_valid_mae_list = [0] * k
    k_valid_mse_list = [0] * k
    k_valid_loss_list = [0] * k
    for i in range(k):
        train_loader = data_loader_list[i][0]
        valid_loader = data_loader_list[i][-1]  
        model = Dual_MLP_T_Emb(input_dim_soap=input_dim, input_dim_local_env=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, drop=drop, temp_dim=temp_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()
        model.train()

        dev_mae = 100
        best_r2 = -1e3

        epoch_list = []
        mae_list = [[], []]
        for epoch in range(epoch_num):
            model.train()
            loss_sum = 0
            y_list = []
            out_list = []
            for (X, y) in train_loader:
                # print(t)
                optimizer.zero_grad()
                out = model(X.to(device))
                loss = criterion(out, y.view(-1, 1).to(device))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                y_list.append(y)
                out_list.append(out.detach().cpu().numpy())
            y_np = np.concatenate(y_list, axis=0)
            out_np = np.concatenate(out_list, axis=0)
            train_mae, train_mse, train_r2 = cal_regression_metrics(y_np, out_np)
            loss = loss_sum / len(train_loader)

            if epoch % 10 == 0:
                # print(loss_sum)
                model.eval()
                valid_mae, valid_mse, valid_r2, valid_loss = evaluate(model, valid_loader, "valid", device, criterion, save_dir)
                
                # log epoch, mae
                epoch_list.append(epoch)
                mae_list[0].append(train_mae)
                mae_list[1].append(valid_mae)
                # log MAE, MSE, R2, loss
                print("Epoch {}'s train loss for {}-th fold: {:.4f} MAE: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(
                    epoch, i, loss, train_mae, train_mse, train_r2))
                print("Epoch {}'s valid loss for {}-th fold: {:.4f} MAE: {:.4f}, MSE: {:.4f}, R2: {:.4f}".format(
                    epoch, i, valid_loss, valid_mae, valid_mse, valid_r2))
                print("***************************************************************************************************************")

                if valid_r2 > best_r2:
                    best_r2 = valid_r2
                    k_valid_r2_list[i] = best_r2
                    k_valid_mae_list[i] = valid_mae
                    k_valid_mse_list[i] = valid_mse
                    k_valid_loss_list[i] = valid_loss
                    torch.save(model.state_dict(), f"{save_dir}/fold_{i}_model.pt")
                    print(f"save {i}-th fold model in epoch {epoch}.")

                    mae, mse, r2, loss = evaluate(model, valid_loader, f"valid_{i}", device, criterion, save_dir, plot=True, log=True, epoch=None)
                    mae, mse, r2, loss = evaluate(model, train_loader, f"train_{i}", device, criterion, save_dir, plot=True, log=True, epoch=None)

                plt.clf()
                plt.cla()
                plt.plot(epoch_list, mae_list[0], label="train")
                plt.plot(epoch_list, mae_list[1], label="valid")
                plt.title(f"Best R2 is {best_r2:.4f}")
                plt.legend()
                plt.savefig(f"{save_dir}/fold_{i}_training.png")
    print(f"Average Valid R2 is {np.mean(k_valid_r2_list)}")
    print(f"Average Valid MAE is {np.mean(k_valid_mae_list)}")
    print(f"Average Valid MSE is {np.mean(k_valid_mse_list)}")
    print(f"Average Valid Loss is {np.mean(k_valid_loss_list)}") 

main()




    



