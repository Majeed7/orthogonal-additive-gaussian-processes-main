import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def plot_synthetic(num, data_set):
    base_save_dir = "/root/OAK_shapley/synthetic_models"
    save_dir = os.path.join(base_save_dir, f"expression_{num+1}")

    svr_rfe_iterations = torch.load(os.path.join(save_dir, "svr_rfe_iter.pt"))
    selectKBest_f_reg_iterations = torch.load(os.path.join(save_dir, "slectKBest_f_reg_iter.pt"))
    selectKBest_mutual_iterations = torch.load(os.path.join(save_dir, "slectKBest_mutual_iter.pt"))
    lasso_iterations = torch.load(os.path.join(save_dir, "lasso_iter.pt"))
    oak_shap_iterations = torch.load(os.path.join(save_dir, "oak_shap_iter.pt"))

    stacked_svr_rfe = np.stack(svr_rfe_iterations)
    stacked_selectKBest_f_reg = np.stack(selectKBest_f_reg_iterations)
    stacked_selectKBest_mutual = np.stack(selectKBest_mutual_iterations)
    stacked_lasso = np.stack(lasso_iterations)
    stacked_oak_shap = np.stack(oak_shap_iterations)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.boxplot([stacked_svr_rfe[:, :4].flatten(),
                 stacked_selectKBest_f_reg[:, :4].flatten(),
                 stacked_selectKBest_mutual[:, :4].flatten(),
                 stacked_lasso[:, :4].flatten(),
                 stacked_oak_shap[:, :4].flatten()],
                labels=['SVR RFE', 'SelectKBest F-reg', 'SelectKBest Mutual', 'Lasso', 'OAK SHAP'])
    plt.title('Box Plots comparing ranks for 4 most important features')
    plt.ylabel("Rank")
    plt.grid(True)
    plt.figtext(0.5, 0.4, f'y = {data_set}', ha='center', fontsize=12)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"synthetic_plot_{num+1}.png"))
    plt.close()
    print(f"Plot saved for expression {num+1}")

expressions = [
    " 2 * sin(2 * X1) + max(X2, 0) + X3 + exp(-X4)",
    "5 * cos(X1 + X2) + 5 * X2 * X3 * X4 + exp(-X4)",
    "8 * X1 + 2 * X1 * X2 + 4 * X1 * X2 * X3 * X4",
    "5 * X1 + 3 * X2 * X3 * X4 + exp(-X4)"
]

for e in range(len(expressions)):
    plot_synthetic(e, expressions[e])