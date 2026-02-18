# Fabric-QC: Automated Fabric Inspection with DCGAN-Augmentation

![Fabric Defect Detection](https://img.shields.io/badge/Project-Fabric--QC-blueviolet?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge)

**Fabric-QC** is an end-to-end deep learning pipeline designed to solve class imbalance in fabric defect detection. By using a **DCGAN** to synthesize minority class samples, the system balances the dataset, allowing a **ResNet50** classifier to achieve significantly higher Macro F1-scores across all defect types.

---

## 🧪 The Pipeline

1.  **Exploratory Data Analysis**: Analyzed a 16.5:1 class imbalance ratio.
2.  **60/20/20 Stratified Split**: Ensured robust evaluation with completely unseen test data.
3.  **Baseline Development**: Established performance using a weighted ResNet50 model.
4.  **DCGAN Implementation**: Trained custom Generators and Discriminators for each defect class (Stain, Hole, etc.).
5.  **Hybrid Training**: Augmented the training set with 2x synthetic samples per defect class.
6.  **Statistical Validation**: Confirmed improvements using **Bootstrap testing** with 1000 iterations.

---

## 🛠 Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **GAN Architecture**: DCGAN (Deep Convolutional GAN) with Binary Cross Entropy Loss
- **Classifier**: ResNet50 with custom classification head
- **Data Handling**: Scikit-learn (GroupShuffleSplit/Stratified), Pandas, Kagglehub
- **Deployment**: FastAPI (Backend) & React (Frontend)

---

## ⚙️ Installation & Usage

### 1. Model Training
Run the `gan-fabric-model.ipynb` notebook. It is configured to:
- Download the `nexuswho/fabric-defects-dataset` automatically.
- Preprocess and split data into `data/processed/`.
- Train the DCGANs and save models to `gan_models/`.
- Export the final hybrid classifier to `models/best_gan_model.pth`.

### 2. Backend Server
```bash
cd backend
pip install fastapi uvicorn torch torchvision Pillow
python app.py
```

### 3. Frontend UI
```bash
cd frontend
npm install
npm start
```

---

## 📊 Performance Metrics

The integration of GAN-generated data consistently improves the **Macro F1-Score**, particularly for the most infrequent defect types.

| Metric | Baseline | GAN-Augmented |
| :--- | :---: | :---: |
| Overall Accuracy | ~92% | **~96%** |
| Macro F1-Score | 0.74 | **0.88** |
| Statistical Significance | - | **p < 0.05 (Significant)** |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.