import torch

import utils
from models.model import ConvNet
from evaluate import evaluate

NUM_EPOCHS = 5

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data, test_data = utils.train_test_dataloaders_by_speaker(data_path='data')

    model = ConvNet(10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train(model, train_data, criterion, optimizer, device)
        evaluate(model, test_data, device)

    torch.save(model.state_dict(), 'audio_mnist_cnn_speakers.pth')

def train(model, train_data, criterion, optimizer, device):
    model.train()
    i = 0
    for inputs, labels, _, _ in train_data:
        inputs = inputs.unsqueeze(1)
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        i+=1
        print(f'{i}/{len(train_data)}')
        print(f"Current loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()