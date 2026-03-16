Bạn là trợ lý kỹ thuật cho đồ án “Drug Solubility Prediction using Graph Neural Networks”.

Bối cảnh:

* Repo: https://github.com/HuyVoTran/drug-solubility-gnn
* Bài toán: dự đoán độ hòa tan trong nước của phân tử (aqueous solubility) dưới dạng regression.
* Input: SMILES của phân tử.
* Target: LogS (logarithm of aqueous solubility).
* Dataset đã có sẵn trong repository: `curated-solubility-dataset.csv`
* Biểu diễn dữ liệu: molecular graph (atoms = nodes, bonds = edges).
* Model bắt buộc: Graph Attention Network (GAT).
* Framework: PyTorch + PyTorch Geometric.
* Mục tiêu: code chạy ổn định trên local và Google Colab, train/test đầy đủ.

Yêu cầu thực hiện:

1. Rà soát và chuẩn hóa pipeline

* scripts/train.py: huấn luyện mô hình
* scripts/test.py: đánh giá trên test set
* train.py ở root chỉ là wrapper gọi scripts/train.py

Pipeline chuẩn cần có:
SMILES → RDKit → molecular graph → PyTorch Geometric Data → GAT model → regression output (LogS)

2. Tiền xử lý dữ liệu

* Đọc dataset `curated-solubility-dataset.csv`
* Chỉ giữ các cột cần thiết:

  * SMILES
  * Solubility (LogS)
* Dùng RDKit để chuyển SMILES → molecular graph
* Node features nên bao gồm:

  * atom type
  * atom degree
  * formal charge
  * hybridization
  * aromaticity
* Edge features:

  * bond type

Chia dataset:

* train: 80%
* validation: 10%
* test: 10%

3. Cấu hình huấn luyện để tránh overfitting

* Optimizer: Adam
* learning_rate mặc định: 1e-3
* weight_decay mặc định: 5e-4
* epochs mặc định: 100
* early stopping (patience = 10)
* batch_size mặc định: 32
* hidden_dim mặc định: 64
* số GAT layers: 2–3
* số attention heads: 4
* dropout mặc định: 0.2

Loss function:

* Mean Squared Error (MSE)

Evaluation metrics:

* RMSE
* MAE
* R² score

Không được fake metric hoặc chỉnh tay loss/metric.

4. DataLoader và hiệu năng

* Sử dụng PyTorch Geometric DataLoader
* batch graph đúng chuẩn
* shuffle cho train set
* pin_memory nếu có CUDA
* num_workers hợp lý để tránh lỗi trên Colab

5. Visualization bắt buộc bằng matplotlib
   Vẽ và lưu các biểu đồ sau:

* training_loss.png
  Train Loss vs Validation Loss

* prediction_vs_true.png
  Scatter plot: True LogS vs Predicted LogS

* residual_plot.png
  Residuals vs Predicted

Yêu cầu:

* biểu đồ rõ ràng
* trục tự động theo dữ liệu
* chỉ dùng metric thực từ quá trình train/validation

6. Output thư mục

* models/
  lưu model checkpoint + metadata

* results/
  lưu metrics (RMSE, MAE, R2)

* plots/

  * training_loss.png
  * prediction_vs_true.png
  * residual_plot.png

7. Test

* scripts/test.py phải:

  * load model đã train
  * load test set
  * chạy inference
  * tính các metric:

    * RMSE
    * MAE
    * R²
* In kết quả ra console
* Lưu kết quả vào `results/`

8. Chất lượng code

* Không sửa ngoài phạm vi yêu cầu
* Sửa tận gốc nguyên nhân lỗi, không vá tạm
* Code rõ ràng, tách module hợp lý
* Giữ style nhất quán
* Sau khi sửa phải kiểm tra syntax

Deliverables bắt buộc:

* Danh sách file đã sửa + tóm tắt thay đổi chính
* Cấu hình train cuối cùng:

  * epochs
  * learning_rate
  * weight_decay
  * hidden_dim
  * GAT heads
  * batch_size
* Lệnh chạy local và Google Colab
* Giải thích ngắn vì sao cấu hình giúp mô hình tổng quát hóa tốt hơn
* Không tạo số liệu giả

Lệnh mẫu cần hỗ trợ:

Local:

* python train.py
* python test.py

Google Colab:

* !git pull
* !python train.py
* !python test.py

git add .
git commit -m "Initial commit - drug solubility GNN project"
git push

!git clone https://github.com/HuyVoTran/drug-solubility-gnn
%cd drug-solubility-gnn
!pip install -r requirements.txt
!python train.py
!python test.py