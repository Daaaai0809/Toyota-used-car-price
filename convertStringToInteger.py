import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# データ読み込み
file_name = 'toyota-integered.csv'
data = pd.read_csv(file_name)
keys = data.keys()      # キー(項目)の名前を取得

# データのコピー
data_new = data.copy()

# pandas形式からリスト形式に変換
att  = keys[5]          # 変換したいキー(項目)を指定 <<<<<<<<<< ここでは2列目(0から数えて)を変換している。
data_pds = data[att]    # extract a column
data_lst = list(data_pds.values)

# データの確認
le = LabelEncoder()     # Label Encoder Obj
le.fit(data_lst)
print('\nTotal {0} types.'.format(len(le.classes_)))

# テキストから数値に変換
data01 = le.transform(data_lst)

# 表示
print('[ ID : `{0}` ]'.format(att))
for i, name in enumerate(le.classes_):
  print(' - {0}: {1}'.format(i, name))

# データの上書き
data_new[att] = data01

# データの保存
data_new.to_csv('toyota-integered.csv', index=False)