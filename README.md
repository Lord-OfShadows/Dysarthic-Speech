# Dysarthria / ALS Speech Detection

This project explores binary dysarthria detection from speech using the TORGO audio dataset. It fine-tunes a small number of parameters on top of Facebook's pretrained HuBERT base model with Low-Rank Adaptation (LoRA), rather than updating the full audio backbone.

The project accompanies [`DL_Report.pdf`](DL_Report.pdf) and is implemented in Google Colab-oriented Jupyter notebooks.

## Project Overview

The classifier predicts one of two labels:

- `0`: control speech
- `1`: dysarthric speech

The dataset loader scans WAV files under folders whose paths contain `con` or `dys`. The current dataset snapshot contains **17,635 recordings**:

| Class | Recordings |
|---|---:|
| Control | 11,456 |
| Dysarthric | 6,179 |

To reduce speaker leakage, the notebooks use speaker-independent splits rather than randomly assigning individual recordings:

| Split | Recordings | Control | Dysarthric |
|---|---:|---:|---:|
| Train | 11,588 | 7,100 | 4,488 |
| Validation | 2,529 | 2,261 | 268 |
| Test | 3,518 | 2,095 | 1,423 |

The test speakers are held out from training. The notebooks identify the intended test speakers as M01, M05, FC01, and MC04, with F01 and FC02 used for validation.

## Approach

1. Download and unpack the TORGO audio data.
2. Load WAV files and infer the binary label from the directory name.
3. Resample audio to **16 kHz** and pad or truncate each example to **3 seconds**.
4. Extract representations with `facebook/hubert-base-ls960`.
5. Apply mean-and-standard-deviation temporal pooling to HuBERT's hidden states.
6. Classify the pooled representation with a small feed-forward head.
7. Evaluate at both recording and speaker level.

The HuBERT backbone is mostly frozen. LoRA adapters are injected into the attention `q_proj` and `v_proj` layers, which keeps the trainable parameter count below 1% of the full model while allowing the representation to adapt to dysarthric speech.

## Anti-Overfitting Edition

[`dysarthria_hubert_lora_antioverfitting.ipynb`](dysarthria_hubert_lora_antioverfitting.ipynb) contains the more heavily regularized experiment described in the report. It includes:

- LoRA rank `r=8`, alpha `16`, and dropout `0.2`
- Classifier-head dropout of `0.6`
- Weight decay of `0.05`
- Label smoothing with `epsilon=0.1`
- Time stretch, pitch shift, additive noise, and time masking
- Learning-rate warmup over 15% of training steps
- Early stopping based on validation AUC-ROC with patience `5`
- A validation-selected decision threshold
- AUC-ROC, PR-AUC, MCC, F1, accuracy, confusion matrix, and per-speaker metrics

## Notebook Variants

- [`dysarthria_hubert_lora_antioverfitting.ipynb`](dysarthria_hubert_lora_antioverfitting.ipynb): regularized experiment with expanded metrics and augmentation.
- [`dysarthria_hubert_lora_antioverfitting (1).ipynb`](dysarthria_hubert_lora_antioverfitting%20(1).ipynb): duplicate/exported copy of the anti-overfitting notebook.
- [`dysarthria_hubert_lora_fixed (1).ipynb`](dysarthria_hubert_lora_fixed%20(1).ipynb): compatibility-focused version with the simpler LoRA configuration (`r=4`, alpha `8`, dropout `0.1`).
- [`DL_Report.pdf`](DL_Report.pdf): project report and discussion.

## Running the Notebooks

The notebooks are designed for Google Colab with a GPU runtime, especially a T4. The fixed notebook pins the main compatibility-sensitive packages:

```bash
pip install transformers==4.40.2 peft==0.10.0 accelerate librosa soundfile scikit-learn matplotlib seaborn
```

Recommended workflow:

1. Open a notebook in Google Colab.
2. Select a GPU runtime.
3. Restart the runtime after installing the packages.
4. Run the cells from top to bottom.
5. Provide Kaggle credentials if the dataset download cell requests them.

The dataset download cell uses the Kaggle dataset [`pranaykoppula/torgo-audio`](https://www.kaggle.com/datasets/pranaykoppula/torgo-audio). Dataset access and licensing terms should be checked on Kaggle before redistribution or reuse.

## Generated Artifacts

Depending on the notebook and completed cells, the pipeline can generate:

- `dysarthria_lora_final.pt`: LoRA adapter weights
- `dysarthria_head_final.pt`: classifier-head weights
- `model_meta.json`: model configuration, threshold, and split metadata
- `evaluation_dashboard.png`: evaluation plots
- CSV/JSON metric exports, including per-speaker results

These artifacts are generated in the notebook runtime and are not included in this repository by default.

## Important Limitations

This is an experimental research pipeline, not a clinical diagnostic system. The dataset is relatively small and speaker-specific effects can strongly influence results, which is why the speaker-independent split and per-speaker reporting are important. Results should be interpreted alongside the report, class balance, held-out speaker identities, and the selected operating threshold.
