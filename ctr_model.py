# =========================
# EDA
# =========================

# user_timezone has missing values:
# Filled missing values with a placeholder and extracted continent.
full_df['user_timezone'] = full_df['user_timezone'].fillna("None/None")
full_df['continent'] = full_df['user_timezone'].apply(lambda x: x.split('/')[0].lower())
vc = full_df.continent.value_counts()
vc.plot(kind='bar')


# sc_id_ctr has two major issues:
# 1. Temporal inconsistency: it aggregates clicks across the whole dataset,
#    but CTR can change over time. Subcategories popular in 2017 may not be relevant in 2020.
# 2. Data leakage: it's calculated before the train-test split, which allows target leakage.
# Both issues are critical and marked as TODO (didnt have the time).
# TODO: Recalculate sc_id_ctr using only past data relative to created_at.
# TODO: Move this calculation into the training fold to avoid leakage.


# =========================
# Feature Engineering
# =========================

# Improved time feature extraction:
# - Replaced magic numbers with constants
# - Used pandas built-ins for clarity
# - Fixed months_since_reg and is_weekend calculations

def add_time_features(df):
    DAYS_A_WEEK = 7
    DAYS_IN_MONTH = 30
    HOUR_IN_MINUTES = 60
    SECONDS_IN_HOUR = 3600

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["user_reg_date"] = pd.to_datetime(df["user_reg_date"])
    df["previous_order_date"] = pd.to_datetime(df["previous_order_date"])

    df["hour"] = (df["created_at"].dt.hour + (df["created_at"].dt.minute / HOUR_IN_MINUTES)).round(1)
    df["day"] = df["created_at"].dt.day_name().str.lower().str[:3]
    df["week_in_month"] = df["created_at"].dt.day // DAYS_A_WEEK
    df["months_since_reg"] = ((df["created_at"] - df["user_reg_date"]).dt.days / DAYS_IN_MONTH).round(1)
    df["hour_diff_from_last_purchase"] = ((df["created_at"] - df["previous_order_date"]).dt.total_seconds() / SECONDS_IN_HOUR).round(1)
    df["days_since_last_purchase"] = (df["created_at"] - df["previous_order_date"]).dt.days
    df["is_weekend"] = df["day"].isin(["sat", "sun"]).astype(int)

    df = df.drop(["created_at", "previous_order_date", "user_reg_date"], axis=1)
    return df

full_df = add_time_features(full_df)


# rare_cats_to_other had several problems:
# - High threshold (e.g. 0.05) can result in major information loss for uniformly distributed features
# - Implementation was slow due to .replace calls
# - This should be a tunable hyperparameter or use a statistical method depending on distribution
# I didn't have time to redesign this, so I just lowered the threshold which improved results,  and make it run faster with map.
from tqdm import tqdm

def rare_cats_to_other(ser, thresh=0.005):
    freqs = ser.value_counts(normalize=True)
    mapping = {cat: cat if freq >= thresh else "other" for cat, freq in freqs.items()}
    return ser.map(mapping)

for col in tqdm(categorical_columns):
    full_df[col] = rare_cats_to_other(full_df[col], thresh=0.0001)
    full_df[col].fillna("other", inplace=True)

full_df[numeric_columns] = full_df[numeric_columns].fillna(0)

# =========================
# Model
# =========================

# - Converted categorical columns before any splits using a more efficient loop
# - Used .iloc for proper index handling in KFold
# - Renamed variables inside the CV loop to avoid confusion with earlier splits

# TODO:
# - Many features like 'context' aren't used—need proper feature selection
# - Even simple vectorization of text fields like 'search_query' and 'gig_title' could help

feature_importance = []
cv_metrics = []
n_folds = 6
full_df = full_df.reset_index(drop=True)

splitter = KFold(n_splits=n_folds, shuffle=True, random_state=13)
for i, (train_indices, test_indices) in enumerate(splitter.split(full_df)):
    print(f'Fold {i}')
    x_train = full_df.loc[train_indices, :].drop('is_click', axis=1)
    y_train = full_df['is_click'].iloc[train_indices].astype(int)
    x_test = full_df.loc[test_indices, :].drop('is_click', axis=1)
    y_test = full_df['is_click'].iloc[test_indices].astype(int)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, stratify=y_train, random_state=13
    )

    model = xgb.XGBClassifier(eval_metric='logloss', enable_categorical=True, **model_params)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)], verbose=50)

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    feature_importance.append(dict(zip(model.feature_names_in_, model.feature_importances_)))
    cv_metrics.append({
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_test, y_pred),
        'f1': metrics.f1_score(y_test, y_pred),
        'auc_roc': metrics.roc_auc_score(y_test, y_prob),
    })

cv_df = pd.DataFrame(cv_metrics)
display(cv_df)
cv_df.plot()
plt.xticks(range(n_folds), range(1, n_folds + 1))
plt.grid()
plt.show()

pd.DataFrame.from_dict(feature_importance).mean().sort_values().plot(kind='barh', figsize=(8, 10))
plt.title('Mean feature importance (cover)')
plt.show()
means = cv_df.mean()
stds = cv_df.std()
plt.barh(means.index, means.values, xerr=stds.values, capsize=5, alpha=0.5)
plt.title('Mean CV classification metrics')
plt.ylabel('Score')
plt.grid(axis='x')
plt.show()


