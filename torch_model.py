
import torch
import torch.nn as nn


class Dual_MLP_T_Emb(nn.Module):
    def __init__(self, input_dim_soap, input_dim_local_env, hidden_dim, output_dim, drop, temp_dim):
        super(Dual_MLP_T_Emb, self).__init__()
        self.soap_dim = input_dim_soap
        self.local_env_dim = input_dim_local_env
        self.temp_dim = temp_dim
        self.feature_extractor_soap = nn.Sequential(
            nn.BatchNorm1d(input_dim_soap),
            nn.Linear(input_dim_soap, hidden_dim),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop) # 0.3-17.6
        )
        self.feature_extractor_local_env = nn.Sequential(
            nn.BatchNorm1d(input_dim_local_env),
            nn.Linear(input_dim_local_env, hidden_dim),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop) # 0.3-17.6
        )
        self.temp_embedding = nn.Embedding(1400, self.temp_dim)
        self.fc1 = nn.Linear(hidden_dim * 2 + self.temp_dim, hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        temp = x[:, -1]
        x_soap = x[:, :self.soap_dim]
        x_local_env = x[:, self.soap_dim:self.soap_dim+self.local_env_dim]
        t_emb = self.temp_embedding(temp.long())
        x_soap = self.feature_extractor_soap(x_soap)
        x_local_env = self.feature_extractor_local_env(x_local_env)

        x = torch.cat([x_soap, x_local_env, t_emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x