import torch
import numpy as np

xy = np.loadtxt('animalzoo.txt',delimiter=',',dtype =np.float32)
x_data = xy[:,:-1]
y_data = xy[:,[-1]]

print(x_data.shape,y_data.shape)
x_dim = x_data.shape[1]
nb_classes = len(np.unique(y_data))
print(f'nb_classes: {nb_classes}')

x = torch.from_numpy(x_data)
y = torch.from_numpy(y_data)

Y_one_hot = torch.zeros(y.size()[0],nb_classes)
Y_one_hot.scatter_(1,y.long(),1)

print("one hot",Y_one_hot.data)

softmax = torch.nn.Softmax()
model = torch.nn.Linear(x_dim,nb_classes,bias=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoch in range(201):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred,y.long().view(-1))
    loss.backward()
    optimizer.step()

    prediction = torch.max(torch.softmax(pred,dim=1),1)[1].float()
    correct_prediction = (prediction.data == y.data.reshape(-1))
    accuracy = correct_prediction.float().mean()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}')

