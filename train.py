from mfccdataset import MFCCDataset
import torch
from torch.utils.data import DataLoader, random_split
from models.model import ConvNet

print(torch.cuda.is_available())
device = torch.device("cuda")

full_dataset = MFCCDataset(data_dir='data')
train_size = int(len(full_dataset)*0.8)
test_size = len(full_dataset)-train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=32)


model = ConvNet(10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 3

def train():

    model.train()
    i = 0
    for inputs, labels in train_data_loader:
        inputs = inputs.unsqueeze(1)
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        i+=1
        print(f'{i}/{len(train_data_loader)}')
        print(f"Current loss: {loss.item():.4f}")


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train()

        evaluate(model, test_data_loader, device)

    model_weights = model.state_dict()
    torch.save(model_weights, 'audio_mnist_cnn.pth')