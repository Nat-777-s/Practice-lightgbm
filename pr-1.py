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

# 残りの2種類に0と1のラベルを与える
df_iris_2class['target'] = df_iris_2class['target'] - 1

X = df_iris_2class.drop('target', axis=1)
y = df_iris_2class['target']

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#テストセットと訓練セットに分ける
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

stratifiedkfold = StratifiedKFold(n_splits=5, shuffle=True,  random_state=50)
train_index, val_index = list(stratifiedkfold.split(X_train_all,y_train_all))[0]
X_train_part, X_val = X_train_all.iloc[train_index], X_train_all.iloc[val_index]
y_train_part, y_val = y_train_all.iloc[train_index], y_train_all.iloc[val_index]

# lightgbmの実装
import lightgbm as lgb
from sklearn.metrics import log_loss

# 特徴量と目的変数をlightgbmのデータ構造に変換する
lgb_train = lgb.Dataset(X_train_part, y_train_part)
lgb_eval = lgb.Dataset(X_val, y_val)

# ハイパーパラメータの設定
params = {'objective': 'binary', 'seed': 70, 'verbose': -1, 'metrics': 'binary_logloss'}
num_round = 100

# 学習の実行
# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする
model = lgb.train(params, lgb_train, num_boost_round=num_round,
                  valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval])

# バリデーションデータでのスコアの確認
va_pred = model.predict(X_val)
score = log_loss(y_val, va_pred)
print(f'logloss: {score:.4f}')

lgb.plot_importance(model, figsize=(8,4), max_num_features=2, importance_type='gain')
plt.show()

# 予測
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1) 

from sklearn import metrics
print(metrics.classification_report(y_test, pred))