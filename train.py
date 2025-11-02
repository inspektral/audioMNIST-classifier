import torch

import utils
from models.model import ConvNet
from evaluate import evaluate

import pandas as pd
import tqdm

from config import NUM_EPOCHS, LOG_TRAINING, NUM_CLASSES, MODEL_WEIGHTS


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    train_data, test_data = utils.train_test_dataloaders_by_speaker(data_path='data')

    model = ConvNet(NUM_CLASSES).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logs = []

    if LOG_TRAINING:
        train_accuracy, train_loss = evaluate(model, train_data, device)
        test_accuracy, test_loss = evaluate(model, test_data, device)

        utils.add_log_entry(logs, 0, train_accuracy, train_loss, test_accuracy, test_loss)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train(model, train_data, criterion, optimizer, device)

        if LOG_TRAINING:
            train_accuracy, train_loss = evaluate(model, train_data, device)
            test_accuracy, test_loss = evaluate(model, test_data, device)

            utils.add_log_entry(logs, epoch+1, train_accuracy, train_loss, test_accuracy, test_loss)

    if LOG_TRAINING:
        df = pd.DataFrame(logs)
        df.to_csv('training_log.csv', index=False)

    torch.save(model.state_dict(), MODEL_WEIGHTS)
    print(f"Model saved to {MODEL_WEIGHTS}")


def train(model, train_data, criterion, optimizer, device):
    model.train()

    for inputs, labels, _, _ in tqdm.tqdm(train_data):
        inputs = inputs.unsqueeze(1)
        
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()