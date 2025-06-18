# 🏆 Yelp Rating Prediction – 1st Place (USC DSCI 553 Spring 2025)

This project placed **1st out of 110 students** in the USC DSCI 553 Spring 2025 course competition. The goal was to build a high-accuracy recommendation system using the Yelp dataset and Spark RDD, under academic and platform constraints (e.g., no DataFrames, limited libraries, memory/time limits).

---

## 📚 Overview

The objective was to predict star ratings for (user, business) pairs in a large, filtered Yelp dataset using a custom recommendation system. Unlike traditional collaborative filtering models, this solution relied on **robust content-based techniques**, engineered features, and performance-tuned processing in Spark RDD.

---

## 🔍 Method Summary

The final method evolved from a baseline content-based model in Homework 3 and introduced:

- 🔬 **Extensive Feature Engineering**:
  - Business metadata (stars, attributes, categories)
  - SVD on business categories
  - Check-in, tip, and photo counts
  - Geolocation features (city, state, lat/lon)
  - **Pseudo-user profiles** derived from averages of visited business attributes

- 🧹 **Feature Selection**:
  - Generated a large feature set, followed by **backward elimination** to remove noise
  - Notably, user-centric models **degraded** performance, including:
    - Item-based collaborative filtering
    - Matrix factorization
    - User bias metrics

- 🧠 **Model Performance**:
  - Locally, **CATBoost** achieved an RMSE of ~0.9708 but was not compatible with the submission environment.
  - Final submitted model used a lightweight, RDD-compatible structure while retaining core feature strategies.

---

## 📈 Results (Validation Set)

- ✅ **Final RMSE**: `0.9747`
- ⏱️ **Execution Time**: `20.18 seconds`

### 📊 Error Distribution

| Error Range   | Count     |
|---------------|-----------|
| 0 ≤ error < 1 | 102,567   |
| 1 ≤ error < 2 | 32,560    |
| 2 ≤ error < 3 | 6,107     |
| 3 ≤ error < 4 | 808       |
| ≥ 4           | 2         |

---

## ⚙️ Technologies & Constraints

- Python 3.6.8
- Spark 3.1.2 (**RDD-only**)
- NumPy, Scikit-learn (no CATBoost on Vocareum)
- Deployed under strict academic constraints (no pretraining, no external datasets
- With more time, fewer computational constraints, and wider access to libraries, my end solution would likely have looked different. I was able to obtain stronger performance locally with the usage of a wider array of features and LightGBM, but this was not feasible in the final submission
---

## ▶️ How to Run

Run this command on the Vocareum platform or compatible Spark 3.1.2 setup:

```bash
/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit competition.py <folder_path> <test_file_name> <output_file_name>
