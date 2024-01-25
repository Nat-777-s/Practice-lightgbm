import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# print(pd.__version__) #スクリプト上でバージョン確認

from sklearn.datasets import load_iris

sk_iris_origin = load_iris()
# print(iris.DESCR)  #データセットに関する説明を表示
# print(type(iris)) #インポートできているか確認

df_iris = pd.DataFrame(sk_iris_origin.data, columns=sk_iris_origin.feature_names)
df_iris['target'] = sk_iris_origin.target

# print(df_iris.shape) #irisデータセットのサイズを確認
# print(df_iris.head()) #データ内容の確認

# #データをプロットする
# fig = plt.figure(figsize = (5,5))
# ax = fig.add_subplot(111)

# #各ラベルごとに花弁の長さをx軸、花弁の幅をy軸に描画
# plt.scatter(df_iris[df_iris['target']==0].iloc[:,0], df_iris[df_iris['target']==0].iloc[:,1], label='Setosa')
# plt.scatter(df_iris[df_iris['target']==1].iloc[:,0], df_iris[df_iris['target']==1].iloc[:,1], label='Versicolour')
# plt.scatter(df_iris[df_iris['target']==2].iloc[:,0], df_iris[df_iris['target']==2].iloc[:,1], label='Virginica')
# plt.xlabel("sepal length[cm]", fontsize=13)
# plt.ylabel("sepal width[cm]", fontsize=13)

# plt.legend()
# plt.show()

#'Versicolour'と'Virginica'の区別に着目
df_iris_2class = df_iris[df_iris['target']!=0]
# print(df_iris_2class.head()) #データ内容の確認

