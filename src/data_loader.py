import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from joblib import dump


def Data_Prepration(path):

       house_dataset = pd.read_csv(path)

       house_dataset_data = house_dataset.iloc[:, 1:]

       label = house_dataset.iloc[:, [0]]

       data_train, data_test, label_train, label_test = train_test_split(house_dataset_data, label, test_size=0.2, random_state=42, shuffle=True)


       # data_train category
       data_train_numerical = data_train.loc[:, ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

       data_train_categorical = data_train.loc[:, ['mainroad',
              'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']]

       data_train_ordinal = data_train.loc[:, ['furnishingstatus']]


       # data_test category
       data_test_numerical = data_test.loc[:, ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

       data_test_categorical = data_test.loc[:, ['mainroad',
              'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']]

       data_test_ordinal = data_test.loc[:, ['furnishingstatus']]


       SS = StandardScaler()

       # as its in the order of alphabet: NO --> 0, YES --> 1
       OE1 = OrdinalEncoder(categories=[['no', 'yes'], ['no', 'yes'], ['no', 'yes'], ['no', 'yes'], ['no', 'yes'], ['no', 'yes']])

       OE2 = OrdinalEncoder(categories=[['unfurnished', "semi-furnished", "furnished"]])

       MMS = MinMaxScaler()

       # Scaling numrrical data
       data_train_numerical_scaled = SS.fit_transform(data_train_numerical)
       data_train_numerical_scaled = pd.DataFrame(data_train_numerical_scaled, columns=data_train_numerical.columns)
       data_test_numerical_scaled = SS.transform(data_test_numerical)
       data_test_numerical_scaled = pd.DataFrame(data_test_numerical_scaled, columns=data_test_numerical.columns)

       # Encoding categorical data
       data_train_categorical_encoded = OE1.fit_transform(data_train_categorical)
       data_train_categorical_encoded = pd.DataFrame(data_train_categorical_encoded, columns=data_train_categorical.columns).astype(int)
       data_test_categorical_encoded = OE1.transform(data_test_categorical)
       data_test_categorical_encoded = pd.DataFrame(data_test_categorical_encoded, columns=data_test_categorical.columns).astype(int)

       # Encoding Ordinal data
       data_train_ordinal_encoded = OE2.fit_transform(data_train_ordinal)
       data_train_ordinal_encoded = pd.DataFrame(data_train_ordinal_encoded, columns=data_train_ordinal.columns).astype(int)
       data_test_ordinal_encoded = OE2.transform(data_test_ordinal)
       data_test_ordinal_encoded = pd.DataFrame(data_test_ordinal_encoded, columns=data_test_ordinal.columns).astype(int)

       # Scaling label
       label_train_scaled = MMS.fit_transform(label_train)
       label_train_scaled = pd.DataFrame(label_train_scaled, columns=label_train.columns)
       label_test_scaled = MMS.transform(label_test)
       label_test_scaled = pd.DataFrame(label_test_scaled, columns=label_test.columns)
       max_price = label_train.max()['price']
       print(max_price)
       min_price = label_train.min()['price']
       print(min_price)
       price_range = max_price - min_price
       print(price_range)

       mid_price = label_train.median()['price']
       print(mid_price)

       dump(SS, '.\\models\\standard_scaler_numerical.pkl')
       dump(OE1, '.\\models\\ordinal_encoder_category.pkl')
       dump(OE2, '.\\models\\ordinal_encoder_ordinal.pkl')


       dataset_train_normalized = pd.concat([data_train_numerical_scaled, data_train_categorical_encoded, data_train_ordinal_encoded], axis=1)
       dataset_test_normalized = pd.concat([data_test_numerical_scaled, data_test_categorical_encoded, data_test_ordinal_encoded], axis=1)

       return dataset_train_normalized, dataset_test_normalized, label_train_scaled, label_test_scaled, price_range, mid_price




































