import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from dataset import get_train_loader, train_dataset_exc, get_test_loader, test_dataset_exc
from test import test
from model import model, device

def get_criterion():
    return CrossEntropyLoss()

def get_optim(model, lr=0.001):
    return Adam(model.parameters(), lr=lr)

def train(epochs, model, criterion, optim, device, train_loader):
    for epoch in range(epochs):
        model.train()

        training_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            training_loss += loss.item()

        print(f"Epoch: {epoch+1}, Loss: {training_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "../resnet_gender_model.pth")


if __name__ == "__main__":
    num_epochs = 5
    criterion = get_criterion()
    optim = get_optim(model)

    train_loader = get_train_loader(train_dataset_exc)
    test_loader = get_test_loader(test_dataset_exc)

    train(num_epochs, model, criterion, optim, device, train_loader)
    test(model, test_loader, device)