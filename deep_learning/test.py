import numpy as np

X = np.matrix([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4x2 (4=num training examples)
y = np.matrix([[0, 1, 1, 0]]).T  # 4x1
numIn, numHid, numOut = 2, 3, 1;  # setup layers
# initialize weights
theta1 = (0.5 * np.sqrt(6 / (numIn + numHid)) * np.random.randn(numIn + 1, numHid))
theta2 = (0.5 * np.sqrt(6 / (numHid + numOut)) * np.random.randn(numHid + 1, numOut))
# initialize weight gradient matrices
theta1_grad = np.matrix(np.zeros((numIn + 1, numHid)))  # 3x2
theta2_grad = np.matrix(np.zeros((numHid + 1, numOut)))  # 3x1
alpha = 0.1  # learning rate
epochs = 1  # num iterations
m = X.shape[0];  # num training examples

print(theta1.shape, theta2.shape)

def sigmoid(x):
    return np.matrix(1.0 / (1.0 + np.exp(-x)))


# backpropagation/gradient descent
for j in range(epochs):
    for x in range(m):  # for each training example
        # forward propagation
        a1 = np.matrix(np.concatenate((X[x, :], np.ones((1, 1))), axis=1)) # 1,3
        z2 = np.matrix(a1.dot(theta1))  # 1x3 * 3x3 = 1x3
        a2 = np.matrix(np.concatenate((sigmoid(z2), np.ones((1, 1))), axis=1)) # 1,4
        z3 = np.matrix(a2.dot(theta2))
        a3 = np.matrix(sigmoid(z3))  # final output # 1,1

        # backpropagation
        delta3 = np.matrix(a3 - y[x])  # 1x1
        # print(theta2.dot(delta3).shape)  # 4,1
        # print(np.multiply(a2, (1 - a2)).T.shape) # (4,1)
        delta2 = np.matrix(np.multiply(theta2.dot(delta3), np.multiply(a2, (1 - a2)).T))  # 1x4
        # print(delta2.shape) # (4,1)

        # Calculate the gradients for each training example and sum them together, getting an average
        # gradient over all the training pairs. Then at the end, we modify our weights.
        theta1_grad += np.matrix((delta2[0:numHid, :].dot(a1))).T  # Notice I omit the bias delta
        # print(theta1_grad.shape) # (3, 3)
        theta2_grad += np.matrix((delta3.dot(a2))).T  # 1x1 * 1x4 = 1x4
        # print(theta2_grad.shape) # (4, 1)

    # update the weights after going through all training examples
    print(theta1.shape, theta1_grad.shape)
    theta1 += -1 * (1 / m) * np.multiply(alpha, theta1_grad)
    print(theta2.shape, theta2_grad.shape)
    theta2 += -1 * (1 / m) * np.multiply(alpha, theta2_grad)
    # reset gradients
    theta1_grad = np.matrix(np.zeros((numIn + 1, numHid)))
    theta2_grad = np.matrix(np.zeros((numHid + 1, numOut)))
