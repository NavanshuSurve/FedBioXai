import flwr as fl
import torch
import torch_directml
from src.models import FedBioXAI
from src.dataloader import FedBioDataset
from torch.utils.data import DataLoader
import sys
import numpy as np
device = torch_directml.device()

class_weights = torch.tensor([0.62, 1.15, 1.98], dtype=torch.float32).to(device)

class BioClient(fl.client.NumPyClient):

    def __init__(self, subject_id):
        self.subject_id = subject_id

        self.train_loader = DataLoader(
            FedBioDataset(subject_id),
            batch_size=32,
            shuffle=True
        )

        self.model = FedBioXAI().to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0003
        )

        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)
        self.model.train()

        total_loss = 0

        for batch in self.train_loader:

            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        print(f"Subject {self.subject_id} - Weighted Loss: {total_loss/len(self.train_loader):.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
        

if __name__ == "__main__":

    if len(sys.argv) > 1:
        sid = sys.argv[1]
    else:
        sid = 'S2'

    print(f"Starting Federated Client for Subject: {sid}")

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=BioClient(subject_id=sid)
    )