import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# print(pd.__version__) #スクリプト上でバージョンを確認できる。

from sklearn.datasets import load_iris

iris = load_iris()
#print(iris.DESCR)  #データセットに関する説明を表示
# print(type(iris)) #インポートできているか確認

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# print(df_iris.shape) #irisデータセットのサイズを確認
df_iris.head()

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