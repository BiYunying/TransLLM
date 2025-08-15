import os
import json
import numpy as np

folder_path = './result_test/your_test_file'

all_success_rates_pred = []
all_dispatch_costs_pred = []
all_success_rates_real = []
all_dispatch_costs_real = []
all_w_distances = []


penalty_weight = np.array([0.5, 1.5, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0, 1.5])

for file_name in os.listdir(folder_path):
    if not file_name.endswith('.json'):
        continue
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as f:
        data_list = json.load(f)

    for data in data_list:
        empty = np.array(data["empty"], dtype=np.float32)  # [9]
        waiting = np.array(data["waiting"], dtype=np.float32)  # [9]
        demand = np.array(data["demand"], dtype=np.float32)  # [9]
        probs = np.array(data["probs"], dtype=np.float32)  # [9]
        real_prob = np.array(data["real_prob"], dtype=np.float32)  # [9]

        # model prediction
        dispatched_pred = empty[0] * probs
        total_empty_pred = waiting + dispatched_pred
        matched_pred = demand * (dispatched_pred / (total_empty_pred + 1e-6))
        success_rate_pred = np.sum(matched_pred) / (empty[0] + 1e-6)
        dispatch_cost_pred = np.sum(dispatched_pred * penalty_weight) / (empty[0] + 1e-6)

        all_success_rates_pred.append(success_rate_pred)
        all_dispatch_costs_pred.append(dispatch_cost_pred)

        # ground truth
        dispatched_real = empty[0] * real_prob
        total_empty_real = waiting + dispatched_real
        matched_real = demand * (dispatched_real / (total_empty_real + 1e-6))
        success_rate_real = np.sum(matched_real) / (empty[0] + 1e-6)
        dispatch_cost_real = np.sum(dispatched_real * penalty_weight) / (empty[0] + 1e-6)

        all_success_rates_real.append(success_rate_real)
        all_dispatch_costs_real.append(dispatch_cost_real)

        # Wasserstein distance
        probs_norm = probs / (probs.sum() + 1e-6)
        real_prob_norm = real_prob / (real_prob.sum() + 1e-6)
        cdf_p = np.cumsum(probs_norm)
        cdf_q = np.cumsum(real_prob_norm)
        w_dist = np.sum(np.abs(cdf_p - cdf_q))
        all_w_distances.append(w_dist)


avg_succ_pred = np.mean(all_success_rates_pred) *100
avg_cost_pred = np.mean(all_dispatch_costs_pred) *3
avg_succ_real = np.mean(all_success_rates_real) *100
avg_cost_real = np.mean(all_dispatch_costs_real) *3
avg_w_dist = np.mean(all_w_distances)

print(f"[Model Dispatch] Average Order Success Rate: {avg_succ_pred:.4f}, Average Empty Travel Distance: {avg_cost_pred:.4f}")
print(f"[Real Probabilities] Average Order Success Rate: {avg_succ_real:.4f}, Average Empty Travel Distance: {avg_cost_real:.4f}")
print(f"[Model vs Real] Average Wasserstein Distance: {avg_w_dist:.4f}")
