"""
Method Description:
My method heavily relied upon the content-based approach of the model first constructed in task 2.2 of homework 3.
I experimented with a large amount of features locally -- as many as I could create from the data, essentially. I think
used backwards elimination to remove features until model performance stagnated. Surprisingly, many approaches that incorporated
information surrounding users degraded performance, such as:
- item-based collaborative filtering
- matrix factorization
- user review bias (avg(bus_stars - user_stars) of businesses reviewed)
My best performing model used the widest amount of business attributes, and SVD on the categories of the business, as well as
pseudo-user profiles of the averages of all the characteristics of the businesses the users had been to. Paired with
CATBoost, I achieved ~.9708 error on the test set. However, this model took too much memory to work on Vocareum, and CATBoost
was unavailable. Nevertheless, I have incorporated many of the same changes below, most relevant of which are:
- category averages
- robust geographic information
- photo/tip/checkin counts
- pseudo-user profiles
In sum, these additions created a meaningful performance improvement.


Error distribution:
>=0 and <1: 102567
>=1 and <2: 32560
>=2 and <3: 6107
>=3 and <4: 808
>=4: 2

Test RMSE: 0.9747

Total execution time: 20.18 seconds

"""


import sys
import os
import time
import pandas as pd
import json
import pyspark
import numpy as np
import datetime
from xgboost import XGBRegressor
from pyspark import SparkContext
from collections import Counter

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext('local[*]', 'itemModelRecYelp')
sc.setLogLevel("ERROR")

input_train_folder_path = sys.argv[1]
input_train_file_path = os.path.join(input_train_folder_path, "yelp_train.csv")
input_val_file_path = os.path.join(input_train_folder_path, "yelp_val.csv")
val_exists = os.path.exists(input_val_file_path)
input_business_file_path = os.path.join(input_train_folder_path, "business.json")
input_user_file_path = os.path.join(input_train_folder_path, "user.json")
input_checkin_file_path = os.path.join(input_train_folder_path, "checkin.json")
input_photo_file_path = os.path.join(input_train_folder_path, "photo.json")
input_tip_file_path = os.path.join(input_train_folder_path, "tip.json")
input_test_file_path = sys.argv[2]
output_file_path = sys.argv[3]


def encode_categories(categories):
    categories = set(categories.split(", ")) if categories else set()
    return [1 if category in categories else 0 for category in top_categories_broadcast.value]


def safe_bool_attr(attr_name, attributes_dict):
    if attributes_dict is None:
        return 0
    val = attributes_dict.get(attr_name, 'False')
    return 0 if val is None else int(str(val) == 'True')


def calculate_business_density_grid(business_rdd, cell_size_km=1.0):
    """Calculate business density within geographic grid cells"""
    cell_size_deg = cell_size_km / 111.0

    # filter for valid locations
    valid_loc_rdd = business_rdd.filter(
        lambda b: 'latitude' in b and 'longitude' in b and
                  b['latitude'] is not None and b['longitude'] is not None
    )

    # calculate grid cells
    business_grid_rdd = valid_loc_rdd.map(
        lambda b: (
            f"{int(float(b['latitude']) / cell_size_deg)}_{int(float(b['longitude']) / cell_size_deg)}",
            b['business_id']
        )
    )

    # count businesses per grid cell
    grid_counts = business_grid_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()

    # map density back to each business
    density_map = valid_loc_rdd.map(
        lambda b: (
            b['business_id'],
            grid_counts.get(f"{int(float(b['latitude']) / cell_size_deg)}_{int(float(b['longitude']) / cell_size_deg)}",
                            1)
        )
    ).collectAsMap()

    # calculate median for default value
    density_values = list(density_map.values())
    density_median = np.median(density_values) if density_values else 1

    return density_map, density_median


def raw_to_features(line, is_training=True):
    user_id, business_id = line[0], line[1]

    (
        user_avg, user_num_reviews, user_yelping_days, user_fans,
        user_avg_business_rating, user_avg_rc, user_avg_price,
        user_avg_lat, user_avg_lon
    ) = user_features_broadcast.value.get(
        user_id, (
            avg_user_rating, 0, 0, 0,
            avg_business_rating, 0, 2, 0.0, 0.0
        )
    )

    business_avg, business_num_reviews, business_is_open, business_price_range, categories, business_city, attributes, latitude, longitude = business_features_broadcast.value.get(
        business_id, (avg_business_rating, 0, 1, 2, "", "UNKNOWN", {}, 0.0, 0.0))

    business_categories = encode_categories(categories)

    # extract category average ratings
    cats = categories.split(", ") if categories else []
    cat_ratings = [category_avg_ratings_broadcast.value.get(c, avg_business_rating) for c in cats]
    cat_avg_rating = np.mean(cat_ratings) if cat_ratings else avg_business_rating

    # get city average rating
    city_avg = city_avg_ratings_broadcast.value.get(business_city, avg_business_rating)

    # activity metrics -- simple counts
    checkin_count = checkin_counts_broadcast.value.get(business_id, 0)
    photo_count = photo_counts_broadcast.value.get(business_id, 0)
    tip_count = tip_counts_broadcast.value.get(business_id, 0)

    # boolean attributes (mostly unimportant)
    has_tv = safe_bool_attr('HasTV', attributes)

    # density metrics
    density_500m = density_500m_broadcast.value.get(business_id, density_500m_median)
    density_1km = density_1km_broadcast.value.get(business_id, density_1km_median)
    density_5km = density_5km_broadcast.value.get(business_id, density_5km_median)

    features = [
                   # user features b
                   user_avg, user_num_reviews, user_yelping_days, user_fans, user_avg_business_rating,
                   user_avg_rc, user_avg_price, user_avg_lat, user_avg_lon,
                   # business basic features
                   business_avg, business_num_reviews, city_avg,
                   business_is_open, business_price_range,
                   # location features
                   latitude, longitude,
                   # activity metrics
                   checkin_count, photo_count, tip_count,
                   # boolean attributes
                   has_tv,
                   # category avg rating
                   cat_avg_rating,
                   # density metrics
                   density_500m, density_1km, density_5km
               ] + business_categories

    if is_training:
        rating = float(line[2])
        return features + [rating]
    else:
        return features


# load data
train_rdd = sc.textFile(input_train_file_path)
header = train_rdd.first()
train_rdd = train_rdd.filter(lambda line: line != header).map(lambda line: line.strip().split(","))

if val_exists:
    val_rdd = sc.textFile(input_val_file_path)
    val_header = val_rdd.first()
    val_rdd = val_rdd.filter(lambda line: line != val_header).map(lambda line: line.strip().split(","))
    train_rdd = train_rdd.union(val_rdd)


business_json_rdd = sc.textFile(input_business_file_path).map(json.loads)
user_json_rdd = sc.textFile(input_user_file_path).map(json.loads)

# extract business features
business_features_dict = business_json_rdd.map(lambda x: (
    x["business_id"],
    (float(x["stars"]), int(x["review_count"]),
     x.get("is_open", 1),
     int(x["attributes"].get("RestaurantsPriceRange2", 2)) if x.get("attributes") else 2,
     x.get("categories", ""),
     x.get("city", "UNKNOWN"),
     x.get("attributes", {}),
     float(x["latitude"]) if x.get("latitude") is not None else 0.0,
     float(x["longitude"]) if x.get("longitude") is not None else 0.0
    )
)).collectAsMap()
business_features_broadcast = sc.broadcast(business_features_dict)

avg_business_rating = np.mean([feat[0] for feat in business_features_dict.values()])

# group training data by user to calculate average business ratings per user
train_by_user = train_rdd.map(lambda x: (x[0], [(x[1], float(x[2]))])) \
    .reduceByKey(lambda a, b: a + b) \
    .collectAsMap()

# calculate average business stats per user
user_business_stats = {}
for user_id, business_ratings in train_by_user.items():
    business_ids = [br[0] for br in business_ratings]
    stats = [business_features_broadcast.value.get(
        bid, (avg_business_rating, 0, 1, 2, "", "UNKNOWN", {}, 0.0, 0.0))
        for bid in business_ids]

    stars_list = [s[0] for s in stats]
    rc_list = [s[1] for s in stats]
    price_list = [s[3] for s in stats]
    lat_list = [s[7] for s in stats]
    lon_list = [s[8] for s in stats]

    user_business_stats[user_id] = {
        "avg_star": np.mean(stars_list),
        "avg_rc": np.mean(rc_list),
        "avg_price": np.mean(price_list),
        "avg_lat": np.mean(lat_list),
        "avg_lon": np.mean(lon_list)
    }


# find the maximum yelping_since date to use as a reference date
max_date = user_json_rdd.map(
    lambda x: datetime.datetime.strptime(x.get("yelping_since", "2000-01-01"), "%Y-%m-%d")
    if x.get("yelping_since") else datetime.datetime(2000, 1, 1)
).max()

print(f"Maximum yelping_since date: {max_date}")

# extract user features
user_features_dict = user_json_rdd.map(lambda x: (
    x["user_id"],
    (
        float(x["average_stars"]),
        int(x["review_count"]),
        int((max_date - datetime.datetime.strptime(x.get("yelping_since", "2000-01-01"), "%Y-%m-%d")).days)
        if x.get("yelping_since") else 0,
        int(x.get("fans", 0)),
        user_business_stats.get(x["user_id"], {}).get("avg_star", avg_business_rating),
        user_business_stats.get(x["user_id"], {}).get("avg_rc", 0),
        user_business_stats.get(x["user_id"], {}).get("avg_price", 2),
        user_business_stats.get(x["user_id"], {}).get("avg_lat", 0.0),
        user_business_stats.get(x["user_id"], {}).get("avg_lon", 0.0)
    )
)).collectAsMap()

user_features_broadcast = sc.broadcast(user_features_dict)

avg_user_rating = np.mean([feat[0] for feat in user_features_dict.values()])

# calculate city average ratings
city_avg_ratings = business_json_rdd.map(lambda x: (x.get("city", "UNKNOWN"), (float(x["stars"]), 1))) \
    .reduceByKey(lambda a, b: ((a[0] * a[1] + b[0] * b[1]) / max(a[1] + b[1], 1), a[1] + b[1])) \
    .mapValues(lambda x: x[0]).collectAsMap()
city_avg_ratings_broadcast = sc.broadcast(city_avg_ratings)

# process categories
category_rdd = business_json_rdd.map(lambda x: x.get("categories", "")).filter(
    lambda x: x is not None and x.strip() != "").map(lambda x: x.split(", "))
category_counts = category_rdd.flatMap(lambda x: x).countByValue()
TOP_N_CATEGORIES = 30
top_categories = [category for category, _ in Counter(category_counts).most_common(TOP_N_CATEGORIES)]
top_categories_broadcast = sc.broadcast(set(top_categories))

# calculate category average ratings
category_avg_ratings = business_json_rdd.flatMap(
    lambda x: [(cat.strip(), (float(x["stars"]) * x["review_count"], x["review_count"]))
               for cat in (x.get("categories", "") or "").split(',') if cat.strip()]
).reduceByKey(
    lambda a, b: (a[0] + b[0], a[1] + b[1])
).mapValues(
    lambda v: v[0] / max(v[1], 1) if v[1] > 0 else avg_business_rating
).collectAsMap()
category_avg_ratings_broadcast = sc.broadcast(category_avg_ratings)

# load checkin data
checkin_json_rdd = sc.textFile(input_checkin_file_path).map(json.loads)
checkin_counts = checkin_json_rdd.map(
    lambda x: (x["business_id"], sum(x["time"].values()) if "time" in x else 0)
).collectAsMap()
checkin_counts_broadcast = sc.broadcast(checkin_counts)

# load photo data
photo_json_rdd = sc.textFile(input_photo_file_path).map(json.loads)
photo_counts = photo_json_rdd.map(
    lambda x: (x["business_id"], 1)
).reduceByKey(lambda a, b: a + b).collectAsMap()
photo_counts_broadcast = sc.broadcast(photo_counts)

# load tip data
tip_json_rdd = sc.textFile(input_tip_file_path).map(json.loads)
tip_counts = tip_json_rdd.map(
    lambda t: (t["business_id"], 1)
).reduceByKey(lambda a, b: a + b).collectAsMap()
tip_counts_broadcast = sc.broadcast(tip_counts)

# calculate business density at different scales
density_500m, density_500m_median = calculate_business_density_grid(business_json_rdd, cell_size_km=0.5)
density_500m_broadcast = sc.broadcast(density_500m)
density_1km, density_1km_median = calculate_business_density_grid(business_json_rdd, cell_size_km=1.0)
density_1km_broadcast = sc.broadcast(density_1km)
density_5km, density_5km_median = calculate_business_density_grid(business_json_rdd, cell_size_km=5.0)
density_5km_broadcast = sc.broadcast(density_5km)

# build training and test features
training_rdd = train_rdd.map(lambda line: raw_to_features(line, is_training=True))
X_train = np.array(training_rdd.map(lambda x: x[:-1]).collect(), dtype=np.float32)
y_train = np.array(training_rdd.map(lambda x: x[-1]).collect(), dtype=np.float32)

# train the mode
model = XGBRegressor(
    objective="reg:linear",
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    reg_alpha=0.05,
    reg_lambda=0.01,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
print(f"Training RMSE: {rmse_train:.4f}")

# process test data
start_time = time.time()

# process test data
test_rdd = sc.textFile(input_test_file_path)
test_header = test_rdd.first()
test_rdd = test_rdd.filter(lambda line: line != test_header).map(lambda line: line.strip().split(","))

testing_rdd = test_rdd.map(lambda line: raw_to_features(line, is_training=True))
X_test = np.array(testing_rdd.map(lambda x: x[:-1]).collect(), dtype=np.float32)
y_test = np.array(testing_rdd.map(lambda x: x[-1]).collect(), dtype=np.float32)
test_data = test_rdd.collect()

# predict
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 1.0, 5.0)

# error analysis
output_df = pd.DataFrame(test_data, columns=["user_id", "business_id", "actual"])
output_df["actual"] = y_test
output_df["prediction"] = y_pred
output_df["error"] = np.abs(output_df["prediction"] - output_df["actual"])

error_bins = {
    '>=0 and <1': (output_df['error'] < 1).sum(),
    '>=1 and <2': ((output_df['error'] >= 1) & (output_df['error'] < 2)).sum(),
    '>=2 and <3': ((output_df['error'] >= 2) & (output_df['error'] < 3)).sum(),
    '>=3 and <4': ((output_df['error'] >= 3) & (output_df['error'] < 4)).sum(),
    '>=4': (output_df['error'] >= 4).sum()
}
print("\nError distribution:")
for bin_label, count in error_bins.items():
    print(f"{bin_label}: {count}")

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nTest RMSE: {rmse:.4f}")

# save results
output_df[['user_id', 'business_id', 'prediction']].to_csv(output_file_path, index=False)

# stop timer and print total execution time
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")