import torch
import torch.nn as nn

x = torch.tensor([[10.0],[9.0],[3.0],[2.0]])
y = torch.tensor([[90.0],[80.0],[50.0],[30.0]])

model = nn.Linear(1,1,bias=True)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

input_x = torch.tensor([[3.5],[4],[5]])
print(f'input_x: {input_x}, model prediction: {model(input_x)}')