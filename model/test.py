import torch

def test(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = (correct/total)*100

        print(f"Test accuracy: {accuracy:.2f}%")