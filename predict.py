import torch
from models.model import ConvNet
import pandas as pd
import tqdm

import utils

from config import DATA_PATH, RESULTS_PATH, MODEL_WEIGHTS


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = ConvNet(10).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    
    data = utils.predict_dataloader(DATA_PATH)

    predictions = predict(model, data, device)
    df = pd.DataFrame(predictions, columns=['filepath', 'predicted_label'])
    df.to_csv(RESULTS_PATH, index=False)


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _, _, filepath in tqdm.tqdm(data_loader):
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.append((filepath[0], predicted.item()))
    return predictions

if __name__ == "__main__":
    print("Starting prediction...")
    main()
    print(f"Predictions saved to {RESULTS_PATH}")