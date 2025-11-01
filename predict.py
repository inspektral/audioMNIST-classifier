import torch
from models.model import ConvNet
import pandas as pd

import utils

PATH = 'predict'
RESULTS_PATH = 'predictions.csv'

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ConvNet(10).to(device)
    model.load_state_dict(torch.load('audio_mnist_cnn_speakers.pth'))
    
    data = utils.predict_dataloader(PATH)

    predictions = predict(model, data, device)
    df = pd.DataFrame(predictions, columns=['filepath', 'predicted_label'])
    print(df)
    df.to_csv(RESULTS_PATH, index=False)


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _, _, filepath in data_loader:
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.append((filepath, predicted.item()))
    return predictions

if __name__ == "__main__":
    print("Starting prediction...")
    main()
    print(f"Predictions saved to {RESULTS_PATH}")