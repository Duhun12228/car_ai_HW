import torch
import torch.nn as nn

x = torch.tensor([[3.0],[4.5],[5.5],[6.5],[7.5],[8.5],[8.0],[9.0],[9.5],[10.0]])
y = torch.tensor([[8.49],[11.93],[16.18],[18.08],[21.45],[24.35],[21.24],[24.84],[25.94],[26.02]])

model = torch.nn.Linear(1,1,bias=True)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'epoch: {epoch}, loss: {loss.item()}')

print(f"model: y = {model._parameters['weight'].data.numpy()}x + {model._parameters['bias'].data.numpy()}")
