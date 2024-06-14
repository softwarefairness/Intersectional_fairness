import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.utils import normalize_cols, encode_grouping, get_grouping

class CustomDataset:
    def __init__(self, csv_file, sensitive_attrs, device_name, random_seed):
        self.device = torch.device(device_name)
        self.random_seed = random_seed
        self.csv_file = csv_file
        
        # Load your own CSV data
        data = pd.read_csv(csv_file)

        # Separate features and labels
        self.features = data.drop(columns=['Probability']).values
        self.labels = data['Probability'].values

        # Normalize features using the provided normalize_cols function      
        self.features = torch.FloatTensor(
            normalize_cols(self.features)
        ).to(self.device)
        
        # Convert to PyTorch tensors
        self.labels = torch.LongTensor(self.labels).to(self.device)
        
        
        # Create sensitive features
        self.sensitive_attrs = sensitive_attrs.strip().split(',')
        self.create_sensitive_features(data)

        # Split data into training, validation, and test sets
        self._split_data(data)  # Pass data to _split_data function

        self.num_train = len(self.train_data[1])
        self.num_val = len(self.val_data[1])
        self.num_test = len(self.test_data[1])
        self.num_classes = len(torch.unique(self.labels))

    def create_sensitive_features(self, data):
        self.num_sensitive_attrs = len(self.sensitive_attrs)

        self.sensitive_grouping = encode_grouping(self.sensitive_attrs)


        self.sensitive_labels = torch.LongTensor(
            get_grouping(self.csv_file, self.sensitive_attrs, self.sensitive_grouping)
        ).to(self.device)
        
        print(self.sensitive_labels)
        
        self.num_sensitive_groups = self.sensitive_labels.max().item() + 1
        

    def _split_data(self, data):
        np.random.seed(self.random_seed)
        indices = np.arange(len(self.labels))
        train_val_indices, test_indices = train_test_split(
            indices, test_size=0.3, random_state=self.random_seed, shuffle=True
        )
        
        
        print(test_indices)
        # Separate test data
        self.test_data = (
            self.features[test_indices], 
            self.labels[test_indices], 
            self.sensitive_labels[test_indices]
        )
        
        # Save raw test data (without 'Probability' column)
        self.raw_test_data = data.iloc[test_indices]
        

        # Remaining data for training/validation
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.2, random_state=self.random_seed
        )
        
        self.train_data = (self.features[train_indices], self.labels[train_indices], self.sensitive_labels[train_indices])
        self.val_data = (self.features[val_indices], self.labels[val_indices], self.sensitive_labels[val_indices])
        self.test_indices = test_indices

    def create_dataloader(self, train_batch_size):
        self.train_loader = DataLoader(
            TensorDataset(self.train_data[0], self.train_data[1], self.train_data[2]),
            batch_size=train_batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(self.val_data[0], self.val_data[1], self.val_data[2]),
            batch_size=len(self.val_data[1]),
            shuffle=False
        )
        self.test_loader = DataLoader(
            TensorDataset(self.test_data[0], self.test_data[1], self.test_data[2]),
            batch_size=len(self.test_data[1]),
            shuffle=False
        )

    def remove_sensitive_features(self):
        sensitive_indices = [i for i, col in enumerate(self.sensitive_attrs) if col in self.features.columns]
        indices_to_keep = [i for i in range(self.features.shape[1]) if i not in sensitive_indices]
        self.features = self.features[:, indices_to_keep]
        self.features = self.features.to(self.device)
