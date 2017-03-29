import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deep_learning.layers import Layer, InputLayer
from deep_learning.models import Model

raw_data = pd.read_csv('insurance.csv')

# Preprocessing
bmi_scaler = StandardScaler()
raw_data[['bmi']] = bmi_scaler.fit_transform(raw_data[['bmi']].values)
raw_data['sex'] = raw_data['sex'].apply({'male': 1, 'female': 0}.get)
raw_data['smoker'] = raw_data['smoker'].apply({'yes': 1, 'no': 0}.get)

# Region
onehot_smokers = pd.get_dummies(raw_data['region'], prefix='region')
raw_data = pd.concat([raw_data, onehot_smokers], axis=1)
# raw_data = raw_data.drop('region', axis=1)

# Y Labels
expense_scaler = StandardScaler()
raw_data['expenses'] = expense_scaler.fit_transform(raw_data[['expenses']].values)
data_y = raw_data.as_matrix(['expenses'])

# X Labels
data_x = raw_data.drop(['expenses', 'region'], axis=1)
data_x = data_x.as_matrix()

# Split data into train and examples subset
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3)

# print('[TRAIN]')
# print('train_x:', train_x.shape)
# print('train_y:', train_y.shape)
# print('\n[TEST]')
# print('test_x:', test_x.shape)
# print('test_y:', test_y.shape)

# Create Model
model = Model()
model.add(InputLayer(9, 64, activation='sigmoid', batch_input_shape=(None, 9), name='input_layer'))
model.add(Layer(64, 32, activation='sigmoid', name='layer01'))
model.add(Layer(32, 16, activation='sigmoid', name='layer02'))
model.add(Layer(16, 8, activation='sigmoid', name='layer03'))
model.add(Layer(8, 1, activation='sigmoid', name='output_layer'))
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(data_x, data_y, epochs=1)
