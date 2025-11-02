# AudioMNIST Classifier

Small CNN classifier for the AudioMNIST dataset. The repository converts audio to MFCCs and trains a compact convolutional network to classify spoken digits (0–9).

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

## Quick start

1. Install dependencies (pick the correct PyTorch wheel for your CUDA/drivers):

```bash
pip install -r requirements.txt
# Example (CPU-only PyTorch):
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. Train the model:

```bash
python train.py
```

3. Evaluate a saved model:

```bash
python evaluate.py
```

4. Predict on new audio placed in `predict_data/`:

```bash
python predict.py
```

## Notes and tips
- The dataset is split by speaker by default (see `utils.train_test_dataloaders_by_speaker`) so test speakers are unseen during training.
- MFCC extraction and fixed-length padding are handled in `mfccdataset.py` (1s clips, 16 kHz, 20 MFCCs by default).
- If you use GPU, ensure PyTorch is installed with the matching CUDA version. When loading weights on a different device use `torch.load(..., map_location=device)`.
- To speed up data loading on Linux, enable `pin_memory=True` and set `num_workers>0` in the DataLoader (edit `utils.py`).

## Results and artifacts
- Trained weights (examples): `audio_mnist_cnn.pth`, `audio_mnist_cnn_speakers.pth`
- Logs: `training_log.csv` — per-epoch loss and accuracy for train/test
- Predictions: `predictions.csv`

## License
MIT — see `LICENSE`.

If you'd like, I can add a short example showing how to load the model and run a single file prediction from Python code.