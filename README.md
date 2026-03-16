# Drug Solubility Prediction using GAT

Mô hình dự đoán `LogS` (aqueous solubility) từ `SMILES` bằng **Graph Attention Network (GAT)** với **PyTorch Geometric**.

## Pipeline

`SMILES -> RDKit -> molecular graph -> PyG Data -> GAT -> LogS regression`

- Node features: atom type, degree, formal charge, hybridization, aromaticity
- Edge features: bond type
- Split: train/val/test = 80/10/10

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy local

```bash
python train.py
python test.py
```

## Chạy Google Colab

```python
!git pull
!pip install -r requirements.txt
!python train.py
!python test.py
```

## Output

- `models/best_gat_model.pt`: checkpoint tốt nhất theo validation MSE
- `models/train_metadata.json`: lịch sử train/val loss
- `results/test_metrics.json`: RMSE, MAE, R2 trên test set
- `plots/training_loss.png`
- `plots/prediction_vs_true.png`
- `plots/residual_plot.png`
