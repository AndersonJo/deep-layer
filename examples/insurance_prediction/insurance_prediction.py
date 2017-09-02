import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from deep_layer.layers import Layer, InputLayer
from deep_layer.models import Model
from deep_layer.optimizers import Momentum
import pylab as plt

np.random.seed(0)

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
expense_scaler = MinMaxScaler()
raw_data['expenses'] = expense_scaler.fit_transform(raw_data[['expenses']].values)
data_y = raw_data.as_matrix(['expenses'])

# X Labels
data_x = raw_data.drop(['expenses', 'region'], axis=1)
data_x = data_x.as_matrix()
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(data_x)

# Split data into train and examples subset
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3)

# Create Model
model = Model()
model.add(InputLayer(9, 32, activation='sigmoid', batch_input_shape=(2, 9), name='input_layer'))
model.add(Layer(32, 16, activation='sigmoid', name='hidden_layer1'))
# model.add(Layer(16, 8, activation='sigmoid', name='hidden_layer2'))
model.add(Layer(16, 1, activation='sigmoid', name='output_layer'))
model.compile(optimizer=Momentum(lr=0.001), batch=64, loss='mean_squared_error')
model.fit(data_x, data_y, epochs=98, shuffle=False)

y_pred = model.predict(test_x)

print('r^2 score:', r2_score(test_y, y_pred))
# y_pred = expense_scaler.inverse_transform(y_pred)
# y_true = expense_scaler.inverse_transform(test_y)
N = len(y_pred)
x = list(range(N))

plt.scatter(x, test_y, color='#333333', label='y_true')
plt.scatter(x, y_pred, color='red', label='y_pred')
plt.grid()
plt.legend()
plt.savefig('result.png')
