import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(pd.__version__) #スクリプト上でバージョンを確認できる。

from sklearn.datasets import load_iris

iris = load_iris()
#print(iris.DESCR)  #データセットに関する説明を表示
print(type(iris))