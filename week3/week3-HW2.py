import torch
import torch.nn as nn
import numpy as np

data = np.loadtxt('diabetes.txt',delimiter=',',dtype=np.float32)
x_data = data[:,0:-1]
y_data = data[:,[-1]]

x_train_1 = torch.tensor(x_data[:int(len(x_data)/2+1)])
y_train_1 = torch.tensor(y_data[:int(len(x_data)/2+1)])
x_test_1 = torch.tensor(x_data[int(len(x_data)/2+1):])
y_test_1 = torch.tensor(y_data[int(len(x_data)/2+1):])

x_train_2 = torch.tensor(x_data[:int(len(x_data)*0.7)+1])
y_train_2 = torch.tensor(y_data[:int(len(x_data)*0.7)+1])
x_test_2 = torch.tensor(x_data[int(len(x_data)*0.7)+1:])
y_test_2 = torch.tensor(y_data[int(len(x_data)*0.7)+1:])

x_train_3 = torch.tensor(x_data[:int(len(x_data)*0.8)+1])
y_train_3 = torch.tensor(y_data[:int(len(x_data)*0.8)+1])
x_test_3 = torch.tensor(x_data[int(len(x_data)*0.8)+1:])
y_test_3 = torch.tensor(y_data[int(len(x_data)*0.8)+1:])

linear = nn.Linear(8,1,bias=True)
sigmoid = nn.Sigmoid()
model_1 = nn.Sequential(linear,sigmoid)
model_2 = nn.Sequential(linear,sigmoid)
model_3 = nn.Sequential(linear,sigmoid)

optimizer = torch.optim.SGD(model_1.parameters(),lr=0.01)

for epoch in range(1000):
    pred = model_1(x_train_1)
    loss = -(y_train_1*torch.log(pred) + (1-y_train_1)*torch.log(1-pred)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'model: 1 /// epoch:{epoch},loss: {loss.data.numpy()}')
print('-'*50)

for epoch in range(1000):
    pred = model_2(x_train_2)
    loss = -(y_train_2*torch.log(pred) + (1-y_train_2)*torch.log(1-pred)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'model: 2 /// epoch:{epoch},loss: {loss.data.numpy()}')
print('-' * 50)

for epoch in range(1000):
    pred = model_3(x_train_3)
    loss = -(y_train_3*torch.log(pred) + (1-y_train_3)*torch.log(1-pred)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'model: 3 /// epoch:{epoch},loss: {loss.data.numpy()}')

print('-' * 50)

predicted_1 = (model_1(x_test_1) > 0.5).float()
predicted_2 = (model_2(x_test_2) > 0.5).float()
predicted_3 = (model_3(x_test_3) > 0.5).float()

accuracy_1 = (predicted_1 == y_test_1).float().mean()
accuracy_2 = (predicted_2 == y_test_2).float().mean()
accuracy_3 = (predicted_3 == y_test_3).float().mean()

print(f'accuracy 1: {accuracy_1}')
print(f'accuracy 2: {accuracy_2}')
print(f'accuracy 3: {accuracy_3}')




