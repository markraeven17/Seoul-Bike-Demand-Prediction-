# Seoul Bike Demand Prediction using Deep Neural Networks

This project uses a deep learning model built with TensorFlow/Keras to predict whether a bike rental station in Seoul is operational ("Functioning Day") based on historical environmental and seasonal data. The model is trained on the publicly available [Seoul Bike Sharing Demand Dataset](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) from the UCI Machine Learning Repository.

---

## Project Goals

- Perform binary classification (`Functioning Day`: Yes/No)
- Preprocess categorical and numerical features using pipelines
- Train a deep neural network to predict station status
- Evaluate model performance with accuracy, loss curves, and confusion matrix

---

## Technologies Used

- Python 
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## Dataset Overview

- Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand)
- Features:
  - Date, Temperature, Humidity, Wind speed, Rainfall
  - Seasons, Holidays, Hour of day
- Target:
  - `Functioning Day`: whether the station was operating (Yes/No)

---

## Preprocessing Steps

- Combined features and labels using `pandas.concat`
- Removed `Date` column (not used in prediction)
- Applied `StandardScaler` to numerical features
- Applied `OneHotEncoder` to categorical features
- Encoded the target label using `LabelEncoder`
- Split data into 80% training and 20% test sets

---

## Model Architecture
```text
Input Layer  →  Dense(50, relu)
             →  Dense(50, relu)
             →  Dense(50, relu)
             →  Dropout(0.3)
             →  Dense(50, relu)
             →  Dense(50, relu)
             →  Dense(1, sigmoid)

Loss Function: binary_crossentropy
Optimizer: Adam
Callback: EarlyStopping with patience=3
```
---
## Training Results

- Train Accuracy	~99.6%
- Validation Accuracy	~99.5%
- Test Accuracy	~99.7%
- Final Loss	~0.0081
