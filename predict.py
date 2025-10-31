import torch
from models.model import ConvNet

import utils

PATH = 'predict'

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ConvNet(10).to(device)
    model.load_state_dict(torch.load('audio_mnist_cnn.pth'))
    
    data = utils.predict_dataloader(PATH)

    predictions = predict(model, data, device)
    print("Predicted class:", predictions[0])

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _, _, filepath in data_loader:
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.append(predicted.item())
            print(f"File: {filepath[0]} --> Predicted class: {predicted.item()}")
    return predictions

if __name__ == "__main__":
    main()