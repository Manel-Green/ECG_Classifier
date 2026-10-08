# ECG Normal/Abnormal Classifier

A personal learning project: fine-tuning a computer vision model to classify
12-lead ECG images as **Normal** or **Abnormal**, from scratch — data
collection through training through an interactive demo.

This started as a standalone exercise to learn transfer learning and
fine-tuning for medical imaging. A version of the underlying model was later
adapted and contributed to a healthcare startup's clinical decision-support
product; this repo contains the original, personal version of the work.

## What it does

Upload a 12-lead ECG image and the model returns a binary call: **Normal**
or **Abnormal**, with a raw probability. It does not attempt to name a
specific diagnosis (e.g. WPW, RBBB) — it's a screening-level classifier,
not a diagnostic one.

## Method

- **Architecture**: ResNet18, pretrained on ImageNet.
- **Fine-tuning strategy**: two stages.
  1. **Frozen backbone** — every layer frozen except a new binary
     classification head, trained for 5 epochs.
  2. **Partial fine-tuning** — `layer3` and `layer4` unfrozen with a lower
     learning rate (1e-5) than the head (1e-3), for 5 more epochs, keeping
     whichever epoch's checkpoint had the best validation accuracy.
- **Data**: 1,682 photographed 12-lead ECGs (859 Normal, 823 Abnormal),
  stratified 80/20 train/validation split.
- **Loss**: `BCEWithLogitsLoss`, trained with Adam.

Full training notebook: [`training.ipynb`](./training.ipynb).

## Results

On the held-out validation split:

| Metric | Score |
|---|---|
| Accuracy | 85.5% |
| Sensitivity (catching real Abnormal) | 85.5% |
| Specificity (correctly clearing Normal) | 85.5% |

## Dataset

Khan, A.H. & Hussain, M., *"ECG Images dataset of Cardiac and COVID-19
Patients"*, Mendeley Data, V2.
Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload any 12-lead ECG image through the browser UI.

## Honest limitations

- This is a **personal learning project**, not a validated clinical tool,
  and was never intended to be used for real diagnostic decisions.
- Validated only against its own training distribution (clean, photographed
  ECGs matching the training data's format) — not tested against messy
  real-world uploads from different scanners/cameras.
- Binary screening only — no claim is made about identifying the specific
  underlying condition.
