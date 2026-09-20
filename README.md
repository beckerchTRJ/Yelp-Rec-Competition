# Yelp Rating Prediction — 1st Place of 110

**Task:** predict the star rating a user will give a business, on a large filtered Yelp dataset.
**Constraint:** Spark **RDD-only** (no DataFrames), a restricted library list, and hard memory and runtime limits.
**Result:** **1st of 110** in the USC DSCI 553 (Spring 2025) course competition — validation **RMSE 0.9747** in **20 seconds** of runtime.

The counterintuitive finding: every collaborative-filtering approach I tried made the model *worse*. The winning solution uses no collaborative filtering at all — just an XGBoost regressor on carefully engineered features.

**In plain terms:** given a person and a business, guess how many stars that person will give it. This is the same problem behind every "recommended for you" list. Accuracy was scored by how far the guesses landed from the real ratings (lower is better), and 72% of my predictions came within one star. The competition rules required an older, more manual way of working with large data and capped both memory and running time, so the solution had to be efficient as well as accurate.

---

## Approach

**Feature engineering**
- Business metadata: average stars, review count, price range, open status, boolean attributes
- Category-level rating averages (the local best model also used SVD on business categories)
- Check-in, tip, and photo counts per business
- Geolocation: latitude/longitude, city, and a business-density grid (businesses per ~1 km cell)
- User history: average rating, review count, account age, fans
- **Pseudo-user profiles** — for each user, the averages of the attributes of businesses they have reviewed (typical price range, typical rating, typical location)

**Feature selection**
- Generated a deliberately wide feature set, then pruned it with **backward elimination**.

**What didn't work**
- Item-based collaborative filtering, matrix factorization, and user-bias terms all *degraded* validation RMSE when blended in. The engineered content features carried more signal than the user–business interaction matrix.

**Model**
- Final submission: XGBoost regressor (500 trees, depth 6, light L1/L2 regularization, 0.8 row and column subsampling).
- Locally, CatBoost reached RMSE ≈ 0.9708, but it exceeded the grading environment's memory limit and CatBoost was not installed there. With fewer platform constraints the final solution would have used a wider feature set and a stronger booster.

## Results (validation set)

| Metric | Value |
|---|---|
| RMSE | **0.9747** |
| Runtime | 20.18 s |

| Absolute error | Predictions | Share |
|---|---|---|
| 0 – 1 | 102,567 | 72.2% |
| 1 – 2 | 32,560 | 22.9% |
| 2 – 3 | 6,107 | 4.3% |
| 3 – 4 | 808 | 0.6% |
| ≥ 4 | 2 | <0.01% |

## Run it

Built for Python 3.6, Spark 3.1.2, NumPy, pandas, and XGBoost.

```bash
spark-submit competition.py <folder_path> <test_file_name> <output_file_name>
```

`<folder_path>` is the directory containing the Yelp training files (`yelp_train.csv`, `business.json`, `user.json`, `checkin.json`, `photo.json`, `tip.json`). The dataset was provided by the course and is not included here.
