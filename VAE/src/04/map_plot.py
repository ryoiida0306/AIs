import pickle
import os

import matplotlib.pyplot as plt

if __name__ == '__main__':
   
    # 現在の作業ディレクトリのパスを取得
    current_working_directory = os.getcwd()

    # パスから現在のディレクトリ名を取得
    current_folder_name = os.path.basename(current_working_directory)

    save_dir = '../../result/' + current_folder_name
    mapping_dir = os.path.join(save_dir, 'mapping', 'map')

    # Load the data from the qz_labels_map.pkl file
    with open(os.path.join(mapping_dir, 'qz_labels_map.pkl'), 'rb') as f:
        qz_labels_map = pickle.load(f)

    # for i, batch in qz_labels_map:
    #     for j, (param, label) in enumerate(zip(*batch)):
    #         mu, log_var = param
    #         plt.scatter(mu[0], mu[1], c=label)
    # plt.savefig(f'../../result/{current_folder_name}/mapping/map/map_plot.pdf')

    # qz_labels_map = qz_labels_map[0]
    # mu = [value[0] for value in qz_labels_map]
    # log_var = [value[1] for value in qz_labels_map]
    # label = [value[2] for value in qz_labels_map]
    mu = qz_labels_map[0]
    log_var = qz_labels_map[1]
    label = qz_labels_map[2]
    mu = mu.detach().numpy()
    log_var = log_var.detach().numpy()
    label = label.detach().numpy()

    # Create scatter plot for mu
    scatter_mu = plt.scatter(mu[:,0], mu[:,1], c=label, cmap='gist_rainbow', s=5)
    # cbar_mu = plt.colorbar(scatter_mu)
    # cbar_mu.set_label('Label')
    plt.legend(*scatter_mu.legend_elements(), title="Labels")
    plt.savefig(f'../../result/{current_folder_name}/mapping/map/map_plot_mu.pdf')
    plt.clf()  # Clear the current figure

    
    scatter_log_var = plt.scatter(log_var[:,0], log_var[:,1], c=label, cmap='gist_rainbow', s=5)
    # cbar_log_var = plt.colorbar(scatter_log_var)
    # cbar_log_var.set_label('Label')
    plt.legend(*scatter_log_var.legend_elements(), title="Labels")
    plt.savefig(f'../../result/{current_folder_name}/mapping/map/map_plot_log_var.pdf')
    plt.clf()  # Clear the current figure
