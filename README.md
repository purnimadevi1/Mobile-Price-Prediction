# Mobile Price Prediction — Machine Learning Project

**License:** MIT  **Language:** Python  **Framework:** Scikit-learn / Jupyter Notebook  **Model Used:** Random Forest Classifier  **Repo Type:** End-to-End ML System  

An end-to-end machine learning project that predicts the price range of mobile phones based on their specifications.  
The project covers data preprocessing, model training, evaluation, and saving artefacts (`model.pkl`, `pipeline.pkl`) for future deployment or reuse.

---

## 🧠 Project Overview

This project builds a machine learning model that predicts the **price category** of a mobile phone (Low, Medium, High, Very High) using its hardware and performance specifications.  

It demonstrates the entire ML workflow:
- Data loading and cleaning  
- Preprocessing using Scikit-learn’s `Pipeline` and `ColumnTransformer`  
- Model training with multiple algorithms  
- Evaluation and visualization of model performance  
- Saving trained models and preprocessing pipelines for reuse

---

## ⚙️ Features

### 🔹 Data Preprocessing
- Handles missing and inconsistent data  
- One-Hot Encoding for categorical features  
- Feature scaling for numerical attributes  
- Dataset split: 80% training, 20% testing  
- Pipeline structure for efficient transformation and reuse  

### 🔹 Model Training
- Trained and compared several classifiers:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Classifier (SVC)  
  - K-Nearest Neighbors (KNN)
- Hyperparameter tuning using `GridSearchCV`  
- **Best performance achieved with Random Forest Classifier**
- Evaluation metrics: Accuracy, Confusion Matrix, and Classification Report  

### 🔹 Visualization
- Correlation heatmap and feature importance plots  
- Model performance comparison (accuracy bar chart)  
- Sample predictions visualization  

### 🔹 Deployment
- Saved artefacts:
  - `model.pkl` — Trained model  
  - `pipeline.pkl` — Preprocessing pipeline  
- Easy reuse for predicting the price range of new mobile specifications  

---

## 📁 Repository Structure
```bash
Mobile_Price_Prediction/
│
├── codefiles/
│ │
│ ├── model_training.ipynb # Jupyter Notebook for data preprocessing, training, and evaluation
│ ├── preprocessing.py # Data cleaning and transformation script
│ └── visualization.py # Plotting and feature analysis
│
├── datasets/
│ │
│ └── mobile_data.csv # Dataset used for training
│
├── model.pkl # Final trained machine learning model
├── pipeline.pkl # Preprocessing pipeline for new data
├── requirements.txt # Dependencies list
├── LICENSE # MIT License
└── README.md # Project documentation
```
---

## 🧩 Installation & Setup

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
### 2. (Optional) Train the model
```bash
jupyter notebook codefiles/model_training.ipynb
```
### 3. Run prediction using saved model
```bash
import pickle
import pandas as pd

# Load saved artefacts
model = pickle.load(open('model.pkl', 'rb'))
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

# Example data
sample = pd.DataFrame({
    'battery_power': [1500],
    'ram': [2048],
    'mobile_wt': [150],
    'px_height': [800],
    'px_width': [1200],
})

# Transform and predict
processed = pipeline.transform(sample)
prediction = model.predict(processed)
print("Predicted Price Range:", prediction[0])

```
## 🧮 How the Model Works

### **Preprocessing (`pipeline.pkl`)**
- Encodes categorical features (if present)  
- Scales numerical columns (battery, RAM, pixel size, etc.)  
- Handles missing values automatically  
- Outputs clean numerical arrays ready for prediction  

---

### **Model Training**
- The dataset is split into **train/test sets (80/20)**.  
- Each algorithm is evaluated using accuracy metrics.  
- **Random Forest** achieved the highest accuracy and is stored as `model.pkl`.  

---

### **Saving Artefacts**
```python
import pickle

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(preprocessing_pipeline, open("pipeline.pkl", "wb"))
```
## 📊 Example Output

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | 85.2%     |
| Decision Tree       | 88.9%     |
| **Random Forest**   | **91.4%** |
| SVC                 | 89.6%     |
| KNN                 | 86.7%     |

### Predicted Output Example
Input Specs:
{
  "battery_power": 1500,
  "ram": 2048,
  "mobile_wt": 150,
  "px_height": 800,
  "px_width": 1200
}

Predicted Price Range → "Medium"

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE)  file for details.






