# AudioMNIST Classifier

Small CNN classifier for the AudioMNIST dataset. The repository converts audio to MFCCs and trains a compact convolutional network to classify spoken digits (0–9).

See colab notebook: https://colab.research.google.com/github/inspektral/audioMNIST-classifier/blob/main/report.ipynb

## Contents
- `train.py` — training script (uses `utils` for dataloaders)
- `evaluate.py` — evaluate a saved model on the test set
- `predict.py` — run predictions on the files in `predict_data/`
- `mfccdataset.py` — PyTorch Dataset that extracts MFCCs with torchaudio
- `utils.py` — helpers: dataloaders, speaker-wise split, prediction loader
- `models/model.py` — `ConvNet` model definition
- `config.py` — simple config (seed, epochs, etc.)
- `requirements.txt` — project dependencies
- `report.ipynb` — short notebook with plots and examples
- `download-dataset.sh` — script to download the AudioMNIST dataset

## Quick start

1. Install dependencies:


```bash
pip install -r requirements.txt
```

2. Download the AudioMNIST dataset:

```bash
./download-dataset.sh
```

3. Train the model:

```bash
python train.py
```

4. Evaluate a saved model:

```bash
python evaluate.py
```

5. Predict on new audio placed in `predict_data/`:

```bash
python predict.py
```

## Notes and tips
- The dataset is split by speaker by default (see `utils.train_test_dataloaders_by_speaker`) so test speakers are unseen during training.
- MFCC extraction and fixed-length padding are handled in `mfccdataset.py` (1s clips, 16 kHz, 20 MFCCs by default).

## Results
- Trained weights (examples): `audio_mnist_cnn_speakers.pth`
- Logs: `training_log.csv` — per-epoch loss and accuracy for train/test

## License
MIT — see `LICENSE`.
