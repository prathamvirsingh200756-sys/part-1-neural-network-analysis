# part-1-neural-network-analysis
# Part 1 — Customer Churn Neural Network Analysis

Binary classification with a feed-forward neural network to predict customer churn.

## Dataset
`customer_churn_nn.csv` — 2000 rows × 17 columns  
Target: `churn` (1 = churned, 0 = retained) | Churn rate ≈ 1.6 %

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Task Summary

| Task | Description |
|------|-------------|
| 1 | Dataset exploration — shape, dtypes, stats, target distribution |
| 2 | Preprocessing — label encoding, standard scaling, stratified split, class weights |
| 3 | Neural network — 2 hidden layers (ReLU), sigmoid output, binary cross-entropy |
| 4 | Training & evaluation — loss/accuracy curves, confusion matrix, classification report |
| 5 | 6 hyperparameter experiments varying depth, LR, batch size, epochs, activation |
| 6 | Reflection — weights/biases, activations, LR effects, overfitting analysis |

## Model Architecture (Baseline)

```
Input (15 features)
  → Dense(64, ReLU) → Dropout(0.2)
  → Dense(32, ReLU) → Dropout(0.2)
  → Dense(1, Sigmoid)
```

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam (lr = 0.001)  
- **Class imbalance:** Handled via `class_weight = {0: 1.0, 1: 63.0}`

## Key Results

| Configuration | Test Acc | Test AUC |
|---|---|---|
| Baseline | 0.9475 | 0.8103 |
| Deeper Network | 0.9400 | 0.6984 |
| High LR (0.01) | 0.9250 | 0.7242 |
| **Low LR + More Epochs** | **0.8475** | **0.9687** |
| Large Batch (128) | 0.9275 | 0.8934 |
| Tanh Activation | 0.8525 | 0.9150 |

> **Best AUC:** Exp-3 (Low LR = 0.0001, 100 epochs) — demonstrates that a lower learning rate with more training time learns a better probability boundary for the rare churn class.

## Repository Structure

```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── task1_dataset_overview.png
    ├── evaluation_outputs.png
    ├── model_comparison_table.png
    └── model_comparison_table.csv
```
