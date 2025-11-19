# Mobile Price Prediction using Unsupervised Learning
##  About  
This repository contains a machine-learning project developed using Python and Jupyter Notebooks. It demonstrates the process of loading datasets, preprocessing features, training a predictive model, and packaging the trained model and preprocessing pipeline for reuse.

The workflow includes:  
- Loading raw datasets from the `datasets/` folder.  
- Cleaning and transforming features (handling missing values, encoding categorical variables).  
- Splitting data into training and testing sets.  
- Training a model (the serialized model is in `model.pkl`).  
- Building a preprocessing and inference pipeline (serialized in `pipeline.pkl`).  
- Persisting both the `model.pkl` and `pipeline.pkl` for deployment or further inference.

## 📁 Repository Contents  
- `codefiles/` — Jupyter Notebook(s) and Python scripts implementing data processing, model training and evaluation.  
- `datasets/` — Raw input data files used in the project.  
- `model.pkl` — Serialized trained machine-learning model.  
- `pipeline.pkl` — Serialized preprocessing + inference pipeline.  
- `README.md` — Project overview and instructions.

##  Usage  
1. Navigate to the `datasets/` directory and ensure the required input files are present.  
2. Open the Notebook or script in `codefiles/` to review or rerun the training workflow.  
3. After training, use the `pipeline.pkl` to preprocess new data, then apply `model.pkl` for predictions.  
4. Use the trained model and pipeline for inference on fresh data or integrate into a larger application.


