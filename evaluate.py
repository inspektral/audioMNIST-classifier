import torch
from models.model import ConvNet

import tqdm

import utils

from config import MODEL_WEIGHTS, NUM_CLASSES

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data, test_data = utils.train_test_dataloaders_by_speaker(data_path='data')
    model = ConvNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    accuracy, loss = evaluate(model, test_data, device)

    print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating model...")
    
    with torch.no_grad():
        for inputs, labels, _, _ in tqdm.tqdm(test_loader):
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