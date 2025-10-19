# Amazon ML Price Prediction

## Overview
This project predicts product prices based on catalog content, images, and product metadata using a combination of **LightGBM** regression and **SentenceTransformer embeddings**. The pipeline includes extensive **data preprocessing**, **feature engineering**, and **text embeddings** to improve model accuracy.

---

## Features
- **Data Cleaning & Preprocessing**: Handles missing values, extracts product quantities, weights, brand names, categories, flags (premium, organic, discount, warranty, new), and dominant color from catalog content.
- **Text Embeddings**: Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate semantic embeddings from product descriptions.
- **Feature Engineering**: Combines numeric, categorical, and text embeddings into a single feature matrix for modeling.
- **Regression Model**: Trains a **LightGBM** model with hyperparameters optimized for better accuracy.
- **Evaluation**: Uses **SMAPE** for validation to measure prediction accuracy.
- **Test Predictions**: Processes test data with the same pipeline and generates submission-ready CSV predictions.

---

## Tech Stack
- **Languages**: Python  
- **Libraries**: LightGBM, scikit-learn, Pandas, NumPy, Joblib, TQDM, re  
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)

---

## Usage

### Install Dependencies
```bash
pip install lightgbm tqdm scikit-learn joblib sentence-transformers
