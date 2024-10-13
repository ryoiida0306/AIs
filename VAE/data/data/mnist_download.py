from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import pickle

mnist_X, mnist_y = fetch_openml("mnist_784", version=1, data_home=".", return_X_y=True)

X = mnist_X.astype("float32").to_numpy()
y = mnist_y.astype(int).to_numpy()

# 学習用と評価用に分割
# X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=10000)

# pickle形式で保存
data_dir = "F:\\mnist\\"  # 生文字列リテラルを使用

# ディレクトリが存在しない場合は作成
os.makedirs(data_dir, exist_ok=True)

X_path = os.path.join(data_dir, "mnist_X.pkl")
Y_path = os.path.join(data_dir, "mnist_Y.pkl")

os.makedirs(data_dir, exist_ok=True)


try:
    with open(X_path, 'wb') as f:
        pickle.dump(X, f)
    with open(Y_path, 'wb') as f:
        pickle.dump(y, f)
    print("データの保存に成功しました。")
except Exception as e:
    print(f"データの保存中にエラーが発生しました: {e}")


print(X_path)
print(Y_path)