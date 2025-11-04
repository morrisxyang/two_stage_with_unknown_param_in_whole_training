import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from torch import nn
import os

# Solve a single 0/1 knapsack instance with Gurobi
def solve_knapsack_gurobi(weights_np, prices_np, cap):
    """
    Parameters
    - weights_np: np.ndarray of shape (n_items,), item weights for one instance
    - prices_np:  np.ndarray of shape (n_items,), item prices/values for one instance
    - cap:        float or int scalar, knapsack capacity

    Returns
    - sol: np.ndarray of shape (n_items,), dtype=int64, 0/1 selection vector
    """
    n_items = weights_np.shape[0]
    m = gp.Model()
    m.Params.OutputFlag = 0
    x_var = m.addVars(n_items, vtype=GRB.BINARY, name="x")
    m.setObjective(gp.quicksum(prices_np[i] * x_var[i] for i in range(n_items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(weights_np[i] * x_var[i] for i in range(n_items)) <= cap, name="capacity")
    m.optimize()
    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution. Status: {m.status}")
    sol = np.array([int(round(x_var[i].X)) for i in range(n_items)], dtype=np.int64)
    return sol

# Generate and save optimal 0/1 solutions using predicted and true weights/prices
# pred_values: (N*item_num, 2) with columns [price, weight]
# test_instances: (N, item_num, 2) with columns [weight, price]
def generate_and_save_solutions_for_run(pred_values, test_instances,
                                        cap, current_folder_path:str, run_base_rel:str):
    if torch.is_tensor(pred_values):
        pred_values = pred_values.detach().cpu().numpy()

    num_test_instances = test_instances.shape[0]
    n_items = test_instances.shape[1]

    pred_pw = pred_values.reshape(num_test_instances, n_items, 2)
    pred_prices_mat = pred_pw[:, :, 0].astype(float)
    pred_weights_mat = pred_pw[:, :, 1].astype(float)

    true_weights_mat = test_instances[:, :, 0].astype(float)
    true_prices_mat = test_instances[:, :, 1].astype(float)

    sols_pred = np.zeros((num_test_instances, n_items), dtype=np.int64)
    sols_true = np.zeros((num_test_instances, n_items), dtype=np.int64)
    for ii in range(num_test_instances):
        sols_pred[ii] = solve_knapsack_gurobi(pred_weights_mat[ii], pred_prices_mat[ii], cap)
        sols_true[ii] = solve_knapsack_gurobi(true_weights_mat[ii], true_prices_mat[ii], cap)

    run_dir_abs = os.path.join(current_folder_path, run_base_rel)
    pred_sols_path = os.path.join(run_dir_abs, f"predicted_sols_cap{cap}.npy")
    true_sols_path = os.path.join(run_dir_abs, f"true_sols_cap{cap}.npy")
    np.save(pred_sols_path, sols_pred)
    np.save(true_sols_path, sols_true)
    print(f"✓ Saved solutions: {pred_sols_path}")
    print(f"✓ Saved solutions: {true_sols_path}")
    return pred_sols_path, true_sols_path


# Scale NN outputs to desired ranges per-dimension (price, weight)
class PriceWeightScaler(nn.Module):
    def __init__(self, price_min: float = 10.0, price_max: float = 45.0,
                 weight_min: float = 15.0, weight_max: float = 35.0):
        super().__init__()
        self.price_min = price_min
        self.price_range = price_max - price_min
        self.weight_min = weight_min
        self.weight_range = weight_max - weight_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[..., 0] -> price, x[..., 1] -> weight
        y = torch.empty_like(x)
        y[..., 0] = self.price_min + self.price_range * x[..., 0]
        y[..., 1] = self.weight_min + self.weight_range * x[..., 1]
        return y
