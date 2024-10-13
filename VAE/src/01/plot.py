import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
import os

# 現在の作業ディレクトリのパスを取得
current_working_directory = os.getcwd()

# パスから現在のディレクトリ名を取得
ver = os.path.basename(current_working_directory)

def save_tensors_as_pdf(input_tensor, output_tensor, epoch):
    """
    入力テンソルと出力テンソルを1つのPDFに保存する関数
    :param input_tensor: 入力テンソル
    :param output_tensor: 出力テンソル
    :param filename: 保存するPDFファイルの名前
    """

    os.makedirs(f'../../result/{ver}/figure', exist_ok=True)
    filename = f'../../result/{ver}/figure/epoch_{epoch}.pdf'
    if input_tensor.is_cuda:
        input_tensor = input_tensor.cpu()
    if output_tensor.is_cuda:
        output_tensor = output_tensor.cpu()
    
    with PdfPages(filename) as pdf:
        # 入力テンソルのプロット
        plt.figure()
        plt.imshow(input_tensor.numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Input Image')
        pdf.savefig()
        plt.close()
        
        # 出力テンソルのプロット
        plt.figure()
        plt.imshow(output_tensor.detach().cpu().numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Output Image')
        pdf.savefig()
        plt.close()