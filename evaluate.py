import torch
from models.model import ConvNet

import utils

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data, test_data = utils.train_test_dataloaders()
    model = ConvNet(10).to(device)
    model.load_state_dict(torch.load('audio_mnist_cnn.pth'))
    evaluate(model, test_data, device)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels, _, _ in test_loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item() * batch_size

    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return accuracy, avg_loss

if __name__ == "__main__":
    main()