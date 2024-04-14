import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, Dataset

# The incoming data is a combination of numerical and categorical data. Write a class to load the data from the CSV file into a custom Dataset class that can handle both numerical and categorical data.
# Categorical data should be one-hot encoded. The class should allow you to select a target variable from the data

class TabularDataset(Dataset):
    def __init__(self, data, output_col=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------
        data: pandas data frame
          The data frame object for the input data. It must
          contain all the continuous, categorical columns and
          the output column to be predicted.

        cat_cols: List of strings
          The names of the categorical columns in the data.
          These columns will be passed through the embedding
          layers in the model. These columns will also be
          one-hot encoded

        output_col: string
          The name of the output variable column in the data
          provided.
        """

        data = data.dropna(axis='index',subset=[output_col]).reset_index(drop=True)

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y =  np.zeros((self.n, 1))

        self.cat_cols = [data.columns[i] for i in range(len(data.columns)) if data.dtypes.iloc[i] == 'object' and data.columns[i]!=output_col]
        self.cont_cols = [col for col in data.columns
                          if col not in self.cat_cols + [output_col]]


        if self.cont_cols:
            self.cont_X = data[self.cont_cols].fillna(-1).astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            ordinal_enc = OrdinalEncoder()
            self.cat_X = ordinal_enc.fit_transform(data[self.cat_cols].fillna('')).astype(np.int64)
        else:
            self.cat_X =  np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [torch.from_numpy(self.y[idx]), torch.from_numpy(self.cont_X[idx]), torch.from_numpy(self.cat_X[idx])]

# Now that we have a custom Dataset class, we can use it to load the data into a DataLoader. 
# #Write a function that takes in the file path of the CSV file, the batch size, and the target 
# variable name and returns two DataLoader objects. One for the training data and one for the validation data.

def create_data_loaders(file_path, target_col, batch_size):
    data = pd.read_csv(file_path)
    train_data = data.sample(frac=0.5, random_state=0)
    val_data = data.drop(train_data.index)

    train_dataset = TabularDataset(data=train_data, output_col=target_col)
    val_dataset = TabularDataset(data=val_data, output_col=target_col)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Define a model class that takes both the the numeric and categorical data as input and returns a single output value.
# The model should have an embedding layer for the categorical data, followed by a linear layer for the continuous data.

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):
        """
        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        no_of_cont: Integer
          The number of continuous features in the data.

        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        output_size: Integer
          The size of the final output.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.
        """
        super(Model, self).__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                                                             for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                       for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                          for size in lin_layer_dropouts])

        print(emb_dims)

    def forward(self, cont_data, cat_data):
        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = lin_layer(x)
            x = F.relu(x)
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)
        
        if torch.isnan(x).all():
            breken
        return x

# Now write a function that takes in the file path of the CSV file, the target variable name, the batch size, and the number of epochs
# and returns a trained model. The function should determine the number of unique values in each categorical column and use that information to create the model.

def train_model(file_path, target_col, batch_size, epochs):
    data = pd.read_csv(file_path).reset_index(drop=True)
    # Fill missing values with 0 for numerical columns and '' for categorical columns
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna('Empty')
        else:
            data[col] = data[col].fillna(-1)
    
    data = data.dropna(axis='index',subset=[target_col]).reset_index(drop=True)

    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    emb_dims = [(data[col].nunique(), min(50, (data[col].nunique() + 1) // 2)) for col in cat_cols]

    # Need the minus 1 to account for the target column
    model = Model(emb_dims, len(data.columns) - len(cat_cols) -1, [200, 100], 1, 0.04, [0.01, 0.05])
    #Put the model on the GPU
    model = model.to('cuda')

    train_data, validation_data = create_data_loaders(file_path, target_col, batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        print('EPOCH: ' , epoch)
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./Log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            step=0
            for y, cont_x, cat_x in train_data:
                
                if step> 24:
                    break

                cat_x = cat_x
                cont_x = cont_x
                y = y

                cat_x = cat_x.to('cuda')
                cont_x = cont_x.to('cuda')
                y = y.to('cuda')

                optimizer.zero_grad()
                output = model(cont_x, cat_x)

                loss = criterion(output, y)
                
                loss.backward()

                optimizer.step()
                step+=1
                prof.step()
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            print(f'Epoch {epoch+1}, Train Loss: {loss.item()}')

        # Validation
        with torch.no_grad():
            for y, cont_x, cat_x in validation_data:
                cat_x = cat_x
                cont_x = cont_x
                y = y

                cat_x = cat_x.to('cuda')
                cont_x = cont_x.to('cuda')
                y = y.to('cuda')

                output = model(cont_x, cat_x)
                loss = criterion(output, y)

            print(f'Epoch {epoch+1}, Validation Loss: {loss.item()}')

    return model


path = 'pytorch/FirstAttempts/Data/MLB2017to2021.csv'

#with torch.profiler.profile(
#    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
#    record_shapes=True,
#    with_stack=True
#) as prof:
output = train_model(path, 'release_pos_z', 1024, 1)
#    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))






