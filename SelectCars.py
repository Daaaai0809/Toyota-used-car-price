import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def select_cars(car_num):
    if car_num == 0:
        car_name = 'Auris'
    elif car_num == 1:
        car_name = 'Avensis'
    elif car_num == 2:
        car_name = 'Aygo'
    elif car_num == 3:
        car_name = 'C-HR'
    elif car_num == 4:
        car_name = 'Camry'
    elif car_num == 5:
        car_name = 'Corolla'
    elif car_num == 6:
        car_name = 'GT86'
    elif car_num == 7:
        car_name = 'Hilux'
    elif car_num == 8:
        car_name = 'IQ'
    elif car_num == 9:
        car_name = 'Land Cruiser'
    elif car_num == 10:
        car_name = 'PROACE VERSO'
    elif car_num == 11:
        car_name = 'Prius'
    elif car_num == 12:
        car_name = 'RAV4'
    elif car_num == 13:
        car_name = 'Supra'
    elif car_num == 14:
        car_name = 'Urban Cruiser'
    elif car_num == 15:
        car_name = 'Verso'
    elif car_num == 16:
        car_name = 'Verso-S'
    elif car_num == 17:
        car_name = 'Yaris'

    car_file_name = 'csv/toyota-integered-{}.csv'.format(car_name)
    car_data = pd.read_csv(car_file_name)
    if not car_data.empty:
        return car_name
    
    # データ読み込み
    file_name = 'csv/toyota-integered.csv'
    data = pd.read_csv(file_name)
    keys = data.keys()      # キー(項目)の名前を取得

    # データのコピー
    data_new = data.copy()

    models = list(data_new[keys[0]].values)

    for i in range(len(data_new)):
        if models[i] != car_num:
            data_new = data_new.drop(i)

    # データの保存
    data_new.to_csv('csv/toyota-integered-{}.csv'.format(car_name), index=False)

    return car_name