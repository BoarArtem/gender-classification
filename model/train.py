import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dataset import get_train_loader, get_test_loader, test_dataset_exc, train_dataset_exc
from test import test
from model import GenderModel


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(device):
    return GenderModel().to(device)

def get_criterion():
    return CrossEntropyLoss()

def get_optim(model, lr=0.01):
    return Adam(model.parameters(), lr)

def train(epochs, model, device, criterion, optim, train_loader):
    print("Training successfully starting...")

    for epoch in range(epochs):
        training_loss = 0

        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            training_loss += loss.item()

        print(f"Epochs: {epoch+1}, Loss: {training_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), "../gender_model.pth")

if __name__ == "__main__":
    device = get_device()

    num_epochs = 5
    model = GenderModel().to(device)
    criterion = get_criterion()
    optim = get_optim(model)

    train_loader = get_train_loader(train_dataset_exc)
    test_loader = get_test_loader(test_dataset_exc)

    train(num_epochs, model, device, criterion, optim, train_loader)
    test(model, test_loader, device)

