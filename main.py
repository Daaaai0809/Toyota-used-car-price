# transmission | fuelType
# 0: automatic | 0: Diesel
# 1: manual    | 1: Hybrid
# 2: other     | 2: Other
# 3: semi-auto | 3: Petrol

# [ ID : `model` ]
#  - 0:  Auris
#  - 1:  Avensis
#  - 2:  Aygo
#  - 3:  C-HR
#  - 4:  Camry
#  - 5:  Corolla
#  - 6:  GT86
#  - 7:  Hilux
#  - 8:  IQ
#  - 9:  Land Cruiser
#  - 10:  PROACE VERSO
#  - 11:  Prius
#  - 12:  RAV4
#  - 13:  Supra
#  - 14:  Urban Cruiser
#  - 15:  Verso
#  - 16:  Verso-S
#  - 17:  Yaris

from SelectCars import select_cars
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

print("Enter the car name")
print("0:  Auris\n1:  Avensis\n2:  Aygo\n3:  C-HR\n4:  Camry\n5:  Corolla\n6:  GT86\n7:  Hilux\n8:  IQ\n9:  Land Cruiser\n10:  PROACE VERSO\n11:  Prius\n12:  RAV4\n13:  Supra\n14:  Urban Cruiser\n15:  Verso\n16:  Verso-S\n17:  Yaris")
car_num = input("> ")

car_name = select_cars(int(car_num))

print("select Features(split with ',')")
print("0: model\n1: year\n2: price\n3: transmission\n4: mileage\n5: fuelType\n6: tax\n7: mpg\n8: engineSize")

features = input("> ")
features = features.split(',')
features = [int(i) for i in features]

print("cross validation num")
cv = int(input("> "))

with open("csv/toyota-integered-{}.csv".format(car_name), 'r') as file:
    line = file.readline()
    data = np.loadtxt(file, delimiter=',', usecols=(range(0, 9)))

inputs = data[:, features]
outputs = data[:, 2]

# データを訓練データとテストデータに分割する
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.5, random_state=256)

# 多項式特徴量の生成
poly = PolynomialFeatures(degree=9)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# データの標準化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_poly)
X_test_std = scaler.transform(X_test_poly)

# Ridge正則化による線形回帰
regr = linear_model.RidgeCV(cv=cv)
regr.fit(X_train_std, y_train)

# 交差検証法による係数の導出とR2スコアの最適化
scores = cross_val_score(regr, X_train, y_train, cv=cv)  # 5分割交差検証
mean_score = np.mean(scores)

# テストデータで予測
y_pred = regr.predict(X_test_std)

# R2スコアの計算
r2 = r2_score(y_test, y_pred)

name = re.sub(r'\s+', '', car_name)

if not os.path.exists("svg/{}".format(name)):
    os.mkdir("svg/{}".format(name))

# グラフのプロット
plt.title(car_name)
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('All Datas')
plt.ylabel('Price')
plt.savefig("svg/{}/{}.svg".format(re.sub(r'\s+', '', car_name), re.sub(r'\s+', '', car_name)))
plt.legend()
plt.show()

print('Coefficients: \n', regr.coef_[:9])
print('Best Alpha: ', regr.alpha_)
print('R2 Score (Test Data): ', r2)