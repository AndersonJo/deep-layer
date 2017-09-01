# Deep Layer
Deep Learning with Numpy<br>
You can learn how feed-forward and backpropagation works. 

# Example 
The example below predicts insurance cost.<br>
The full code is [here](https://github.com/AndersonJo/deep-layer/blob/master/examples/insurance_prediction/insurance_prediction.py)

```
>> import numpy as np
>> from deep_layer.layers import Layer, InputLayer
>> from deep_layer.models import Model
>> from deep_layer.optimizers import Momentum
>>
>> np.random.seed(0)
>>
>> model = Model()
>> model.add(InputLayer(9, 32, activation='sigmoid', batch_input_shape=(2, 9), name='input_layer'))
>> model.add(Layer(32, 16, activation='sigmoid', name='hidden_layer1'))
>> model.add(Layer(16, 1, name='output_layer'))
>> model.compile(optimizer=Momentum(lr=0.0005), batch=128, loss='mean_squared_error')
>> model.fit(data_x, data_y, epochs=15, shuffle=False)

[Epoch 0] loss: 0.8041550229987101
[Epoch 1] loss: 0.6849386180886644
[Epoch 2] loss: 0.5852776321377485
[Epoch 3] loss: 0.49581906221271954
[Epoch 4] loss: 0.4161674836035164
[Epoch 5] loss: 0.3604390973829448
[Epoch 6] loss: 0.3008422335309807
[Epoch 7] loss: 0.2576607232879699
[Epoch 8] loss: 0.22932737798070446
[Epoch 9] loss: 0.1782664127999551
[Epoch 10] loss: 0.15898536579946815
[Epoch 11] loss: 0.1288886044363849
[Epoch 12] loss: 0.11173969296987162
[Epoch 13] loss: 0.10028934050214187
[Epoch 14] loss: 0.07579451850678974
```

