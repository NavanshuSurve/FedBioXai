import flwr as fl
import numpy as np
import torch
from src.models import FedBioXAI

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_weights = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_weights is not None:
            # Convert to state_dict
            params_dict = zip(FedBioXAI().state_dict().keys(), 
                            fl.common.parameters_to_ndarrays(aggregated_weights[0]))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            
            # SAVE AS A NEW FILENAME (Do not overwrite model_s2.pth)
            filename = f"model_fed_round_{server_round}.pth"
            torch.save(state_dict, filename)
            print(f"✅ Federated Model saved as: {filename}")

        return aggregated_weights

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=6,
    min_available_clients=6,
)

if __name__ == "__main__":
    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=20), strategy=strategy)