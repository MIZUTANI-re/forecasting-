
import random
import copy

class HyperparameterSampler:
    def __init__(self):
        # Common search space
        self.common_space = {
            "learning_rate": [1e-3, 5e-4, 1e-4],
            "d_model": [64, 128, 256],
            "nhead": [2, 4, 8],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.1, 0.2],
            "hidden_dim": [128, 256, 512],
            "batch_size": [32, 64]
        }
        
        # Model-specific spaces
        self.model_spaces = {
            "FiLM": {
                "modes": [32, 64],
                "film_mode": ["bidir", "v2t", "t2v"]
            },
            "htmformer": {
                "nhead_time": [2, 4, 8]
            },
            "iTransformer": {
                # Mostly standard transformer params
            },
            "MMK": {
                "d_model": [64, 128], # MMK is heavy, keep smaller
                "n_experts_base": [2, 4, 8] # Will expand to list
            },
            # Hybrids with CNN
            "Hybrid_Gated_FeatureFusion": {},
            "Hybrid_Gated_QueryFusion": {},
            "Hybrid_KAN_Gated_FeatureFusion": {},
            "Hybrid_Predict_Fusion": {},
            "Hybrid_NoHQ_Predict_Fusion": {},
            "Hybrid_FiLM_Ablation": {"modes": [32, 64]},
            "grid_tst": {}, # GridTST specific? Using standard for now
            "TransformerFixed": {},
            "MLP": {
                "num_layers": [1, 2, 3],
                "hidden_dim": [64, 128, 256] 
            }
        }

    def sample_params(self, model_name):
        """Returns a parameter dictionary for a single trial."""
        params = {}
        
        # 1. Sample Common Params
        for k, v in self.common_space.items():
            params[k] = random.choice(v)
            
        # 2. Sample Model Specific Params
        specific = self.model_spaces.get(model_name, {})
        for k, v in specific.items():
            params[k] = random.choice(v)
            
        # 3. Constraint Formatting
        # Ensure d_model is divisible by nhead
        if "nhead" in params:
            # Adjust d_model to be multiple of nhead
            rem = params["d_model"] % params["nhead"]
            if rem != 0:
                params["d_model"] += (params["nhead"] - rem)
        
        if "nhead_time" in params:
             rem = params["d_model"] % params["nhead_time"]
             if rem != 0:
                params["d_model"] += (params["nhead_time"] - rem)

        return params

    def optuna_sample_params(self, trial, model_name):
        """Returns a parameter dictionary for a single Optuna trial."""
        params = {}
        
        # 1. Sample Common Params
        for k, v in self.common_space.items():
            params[k] = trial.suggest_categorical(k, v)
            
        # 2. Sample Model Specific Params
        specific = self.model_spaces.get(model_name, {})
        for k, v in specific.items():
            params[k] = trial.suggest_categorical(k, v)
            
        # 3. Constraint Formatting
        if "nhead" in params:
            rem = params["d_model"] % params["nhead"]
            if rem != 0:
                params["d_model"] += (params["nhead"] - rem)
        
        if "nhead_time" in params:
             rem = params["d_model"] % params["nhead_time"]
             if rem != 0:
                params["d_model"] += (params["nhead_time"] - rem)

        return params

    def get_fixed_params(self):
        """Returns the baseline fixed parameters."""
        return {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "hidden_dim": 256,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "nhead_time": 4,
            "cnn_kernel": 3,
            "cnn_mode": "linear",
            "modes": 32,
            "film_mode": "bidir"
        }

    def get_optimized_params(self, model_name, season):
        """Returns the best parameters optimized for the 'All' seasons dataset, applied universally."""
        if model_name == "iTransformer":
            # Best 'All' seasons parameters
            return {"d_model": 128, "nhead": 16, "num_layers": 4, "hidden_dim": 256, "dropout": 0.149, "learning_rate": 0.000031, "batch_size": 64, "cnn_kernel": 3, "cnn_mode": "linear", "modes": 32, "film_mode": "bidir"}
            
        elif model_name == "TransformerFixed":
            # Best 'All' seasons parameters
            return {"d_model": 64, "nhead": 8, "num_layers": 2, "num_decoder_layers": 3, "hidden_dim": 512, "dropout": 0.050, "learning_rate": 0.000344, "batch_size": 64, "cnn_kernel": 3, "cnn_mode": "linear", "modes": 32, "film_mode": "bidir"}
            
        elif model_name == "Hybrid_Gated_FeatureFusion":
            # Best 'All' seasons parameters
            return {"d_model": 256, "nhead": 16, "num_layers": 1, "hidden_dim": 1024, "dropout": 0.134, "learning_rate": 0.00019, "batch_size": 32, "cnn_kernel": 1, "cnn_mode": "linear", "modes": 32, "film_mode": "bidir"}
            
        elif model_name == "Hybrid_FiLM_Ablation":
            # Best 'All' seasons parameters
            return {"d_model": 64, "nhead": 8, "num_layers": 1, "hidden_dim": 512, "dropout": 0.138, "learning_rate": 0.00119, "batch_size": 64, "cnn_kernel": 1, "cnn_mode": "linear", "modes": 32, "film_mode": "bidir"}
        
        return self.get_fixed_params()
