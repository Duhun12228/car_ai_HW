import torch
import numpy as np

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x = torch.Tensor(x_data)
y = torch.Tensor(y_data)

nb_classes = 3

softmax = torch.nn.Softmax()
linear = torch.nn.Linear(4,nb_classes,bias=True)
model = torch.nn.Sequential(linear,softmax)

optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for step in range(2001):
    optimizer.zero_grad()
    pred = model(x)
    loss = -y*torch.log(pred)
    loss = torch.sum(loss,1).mean()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(step,loss.data.numpy())
