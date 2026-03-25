import torch
import torchvision
import random
import numpy as np

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

learning_rate = 0.01
training_epochs = 10
batch_size = 32

# 데이터 로딩
mnist_train = torchvision.datasets.MNIST(root='MNIST_data/', train=True,
                                         transform=torchvision.transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST(root='MNIST_data/', train=False,
                                        transform=torchvision.transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)

x_dim = mnist_train.data.shape[1] * mnist_train.data.shape[2]

n_classes = len(np.unique(mnist_train.targets.numpy()))

model = torch.nn.Linear(x_dim, n_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):
    avg_loss = 0
    total_batch = len(mnist_train) // batch_size

    for i, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        x = batch_x.view(-1, x_dim)
        y = batch_y

        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() / total_batch

    print(f'[Epoch: {epoch+1} cost = {avg_loss:.9f}]')

print('Finished Training')

x_test = mnist_test.data.view(-1, x_dim).float().to(device)
y_test = mnist_test.targets.to(device)


pred = model(x_test)
correct_pred = torch.max(pred, 1)[1] == y_test
accuracy = correct_pred.float().mean()

print(f'Accuracy: {accuracy*100:.2f}%')

r = random.randint(0,len(mnist_test)-1)
x_single_data = mnist_test.data[r:r+1].view(-1,x_dim).float().to(device)
y_single_data = mnist_test.targets[r:r+1].to(device)

print(f'Label: {y_single_data}')
single_pred = model(x_single_data)
print(f'Prediction: {torch.max(single_pred.data,1)[1]}')