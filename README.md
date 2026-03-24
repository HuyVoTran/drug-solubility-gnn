<!-- 

git add .
git commit -m "fix feature"
git push

 -->

# Drug Solubility Prediction using Graph Attention Networks

Dự đoán **độ hòa tan trong nước (aqueous solubility)** của phân tử dưới dạng **regression** bằng **Graph Attention Network (GAT)**, sử dụng **PyTorch + PyTorch Geometric**. Dataset sử dụng là **AqSolDB** — một trong những bộ dữ liệu chuẩn cho bài toán dự đoán độ hòa tan phân tử. Phân tử được biểu diễn dưới dạng molecular graph với atoms là nodes và bonds là edges.

---

## Mục lục

1. [Bài toán](#bài-toán)
2. [Cấu trúc project](#cấu-trúc-project)
3. [Pipeline](#pipeline)
4. [Đặc trưng dữ liệu](#đặc-trưng-dữ-liệu)
5. [Kiến trúc model](#kiến-trúc-model)
6. [Cấu hình huấn luyện](#cấu-hình-huấn-luyện)
7. [Cài đặt](#cài-đặt)
8. [Chạy local](#chạy-local)
9. [Chạy trên Google Colab](#chạy-trên-google-colab)
10. [Output và artifacts](#output-và-artifacts)
11. [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
12. [Giải thích cấu hình](#giải-thích-cấu-hình)

---

## Bài toán

| Hạng mục | Mô tả |
|---|---|
| **Task** | Regression |
| **Input** | SMILES string của phân tử |
| **Target** | LogS — logarithm of aqueous solubility |
| **Dataset** | AqSolDB — `curated-solubility-dataset.csv` (~9 983 mẫu) |
| **Model** | Graph Attention Network (GAT) |
| **Framework** | PyTorch + PyTorch Geometric + RDKit |

---

## Cấu trúc project

```
drug-solubility-gnn/
│
├── curated-solubility-dataset.csv   # Dataset gốc
│
├── train.py                         # Wrapper root → gọi scripts/train.py
├── test.py                          # Wrapper root → gọi scripts/test.py
│
├── scripts/
│   ├── train.py                     # Pipeline huấn luyện đầy đủ
│   └── test.py                      # Pipeline đánh giá trên test set
│
├── src/
│   └── drug_solubility_gnn/
│       ├── __init__.py
│       ├── data_utils.py            # SMILES → RDKit → PyG Data
│       ├── model.py                 # GATRegressor
│       └── metrics.py              # RMSE, MAE, R2, Accuracy
│
├── models/
│   ├── best_gat_model.pt            # Checkpoint tốt nhất (val MSE)
│   └── train_metadata.json         # Lịch sử loss/accuracy từng epoch
│
├── results/
│   └── test_metrics.json           # RMSE, MAE, R2 trên test set
│
├── plots/
│   ├── training_loss.png            # Train vs Validation Loss
│   ├── training_accuracy.png        # Train vs Validation Accuracy
│   ├── prediction_vs_true.png       # Scatter: True vs Predicted LogS
│   └── residual_plot.png            # Residuals vs Predicted
│
└── requirements.txt
```

---

## Pipeline

```
SMILES
  └─► RDKit (Chem.MolFromSmiles)
        └─► Molecular Graph
              ├─► Node features (atoms)
              └─► Edge features (bonds)
                    └─► PyTorch Geometric Data object
                          └─► GATRegressor
                                └─► Predicted LogS (scalar)
```

---

## Đặc trưng dữ liệu

### Node features (mỗi atom — vector 27 chiều)

| Feature | Encoding | Kích thước |
|---|---|---|
| Atom type | One-hot (H, C, N, O, F, P, S, Cl, Br, I, Si, B, Se, other) | 14 |
| Atom degree | One-hot (0–5) | 6 |
| Formal charge | Scalar | 1 |
| Hybridization | One-hot (SP, SP2, SP3, SP3D, SP3D2, other) | 6 |
| Aromaticity | Binary | 1 |

### Edge features (mỗi bond — vector 4 chiều)

| Feature | Encoding |
|---|---|
| Bond type | One-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC) |

### Chia dataset

| Split | Tỉ lệ |
|---|---|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

---

## Kiến trúc model

```
Input node features (27-dim)
    │
    ▼
Linear projection → hidden_dim
    │
    ▼
GATConv Layer 1  (heads=4, concat=False, edge_dim=4)
    │
    ▼
GATConv Layer 2  (heads=4, concat=False, edge_dim=4)
    │
    ▼
Global Mean Pooling (graph → vector)
    │
    ▼
MLP: Linear(80) → ReLU → Dropout → Linear(1)
    │
    ▼
Predicted LogS (scalar)
```

- **Activation**: ELU sau mỗi GATConv layer
- **Pooling**: `global_mean_pool` — tổng hợp tất cả node embeddings thành 1 vector đại diện cho toàn phân tử

---

## Cấu hình huấn luyện

| Tham số | Giá trị mặc định |
|---|---|
| `epochs` | 150 |
| `min_epochs_before_stop` | 100 |
| `learning_rate` | 1e-3 |
| `weight_decay` | 2e-4 |
| `batch_size` | 128 |
| `hidden_dim` | 80 |
| `num_layers` | 2 |
| `heads` | 4 |
| `dropout` | 0.2 |
| `early_stopping patience` | 22 |
| `scheduler` | ReduceLROnPlateau (factor=0.5, patience=4, min_lr=1e-5) |
| `loss` | MAELoss (L1) |
| `optimizer` | Adam |
| `accuracy threshold` | 0.5 LogS unit |

---

## Cài đặt

> Yêu cầu Python ≥ 3.10

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:

```
torch>=2.2
torch-geometric>=2.5
rdkit>=2023.9.5
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
tqdm>=4.66
```

> **Lưu ý GPU**: Nếu muốn dùng CUDA, cài `torch` với CUDA wheel phù hợp trước khi `pip install -r requirements.txt`. Pipeline tự phát hiện `cuda` nếu có.

---

## Chạy local

```bash
# Huấn luyện model
python train.py

# Đánh giá trên test set (cần chạy train.py trước)
python test.py
```

Tuỳ chỉnh hyperparameter qua CLI:

```bash
python train.py --epochs 150 --hidden-dim 80 --num-layers 2 --dropout 0.2 --batch-size 128
python test.py --num-runs 5
python test.py --seeds 42 123 2024 7 999
```

---

## Chạy trên Google Colab

```python
# Bước 1: Clone repository
!git clone https://github.com/HuyVoTran/drug-solubility-gnn
%cd drug-solubility-gnn

# Bước 2: Cài PyTorch (CUDA 12.1)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Bước 3: Cài PyTorch Geometric và các thư viện còn lại
!pip install torch-geometric
!pip install rdkit pandas numpy scikit-learn matplotlib tqdm

# Bước 4: Huấn luyện
!python train.py

# Bước 5: Đánh giá thống kê (5 seeds mặc định)
!python test.py
```

Xem kết quả và biểu đồ ngay trong notebook:

```python
import json
from IPython.display import Image, display

# Metrics tổng hợp
with open("results/statistical_test_metrics.json") as f:
    print(json.dumps(json.load(f), indent=2))

# Biểu đồ
display(Image("plots/training_loss.png"))
display(Image("plots/training_accuracy.png"))
display(Image("plots/prediction_vs_true.png"))
display(Image("plots/residual_plot.png"))
```

Nếu đã clone rồi và chỉ muốn cập nhật code mới nhất:

```python
!git pull
!python train.py
!python test.py
```

---

## Output và artifacts

### Sau `python train.py`

| File | Mô tả |
|---|---|
| `models/best_gat_model.pt` | Checkpoint epoch tốt nhất theo validation MSE (bao gồm `model_state_dict`, `config`, `feature_info`, `split_indices`) |
| `models/train_metadata.json` | Lịch sử train/val loss và accuracy từng epoch |
| `plots/training_loss.png` | Biểu đồ Train Loss vs Validation Loss theo epoch |
| `plots/training_accuracy.png` | Biểu đồ Train Accuracy vs Validation Accuracy theo epoch |

### Sau `python test.py`

| File | Mô tả |
|---|---|
| `results/test_metrics.json` | RMSE, MAE, R² trên test set (seed đầu tiên) |
| `results/statistical_test_metrics.json` | RMSE, MAE, R² trung bình ± std trên nhiều seeds |
| `plots/prediction_vs_true.png` | Scatter plot: True LogS vs Predicted LogS |
| `plots/residual_plot.png` | Residuals (True − Predicted) vs Predicted LogS |

---

## Kết quả thực nghiệm

Kết quả đánh giá trên **test set** (10% dataset, ~998 mẫu), checkpoint epoch 84:

| Metric | Giá trị |
|---|---|
| **RMSE** | 1.1497 |
| **MAE** | 0.8173 |
| **R²** | 0.7534 |

> Accuracy (threshold = 0.5 LogS unit): ~45% mẫu test có sai lệch < 0.5 đơn vị LogS.

---

## Giải thích cấu hình

| Quyết định | Lý do |
|---|---|
| `hidden_dim=64` + `num_layers=2` | Capacity nhỏ gọn, giảm overfitting, val loss dễ theo kịp train loss |
| `dropout=0.25` | Regularization cân bằng — đủ để giảm gap train/val mà không làm model underfitting |
| `weight_decay=3e-4` | L2 penalty vừa đủ để tổng quát hóa tốt trên phân tử chưa thấy |
| `batch_size=128` | Gradient mượt hơn, giảm oscillation đáng kể so với batch nhỏ |
| `learning_rate=5e-4` | Khởi đầu chậm hơn, tránh spike đầu training, loss giảm đều hơn |
| `ReduceLROnPlateau` (patience=8, min_lr=1e-5) | Không giảm LR quá sớm; `min_lr` tránh model bị đông cứng |
| `early stopping` (patience=20, min=100) | Đảm bảo chạy ít nhất 100 epoch, sau đó dừng nếu val loss không cải thiện 20 epoch liên tiếp |
| `L1Loss (MAE)` | Robust hơn với outliers so với MSE, loss thực tế phản ánh đúng sai số tuyệt đối |
| `heads=4` | Multi-head attention học nhiều loại tương tác hóa học khác nhau cùng lúc |
| `edge_attr` (bond type) | Thêm thông tin loại liên kết giúp GAT phân biệt cấu trúc tốt hơn |
| `global_mean_pool` | Pooling bền vững với phân tử có kích thước khác nhau |
| EMA smoothing (α=0.15) | Đường loss/accuracy mượt hơn trên biểu đồ, xu hướng hội tụ rõ ràng hơn |
