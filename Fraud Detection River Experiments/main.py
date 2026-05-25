import random
from collections import defaultdict
from river import metrics, stats, compose, preprocessing, linear_model

user_mean = defaultdict(stats.Mean)
user_var = defaultdict(stats.Var)
user_count = defaultdict(int)

# ---------------------------
# SYNTHETIC STREAM
# ---------------------------
def stream_transactions(n=10000):
    for i in range(n):
        user_id = random.randint(1, 50)

        base_mean = 30 + (user_id % 10) * 5

        # IMPORTANT: snapshot state BEFORE update
        past_count = user_count[user_id]

        amount = random.expovariate(1 / base_mean)

        amount_anomaly = amount / base_mean

        velocity_anomaly = 1 if past_count > 8 else 0

        global_drift = 0.01 if i < 3000 else 0.08

        fraud_prob = (
            global_drift
            + 0.35 * max(0, amount_anomaly - 2)
            + 0.25 * velocity_anomaly
        )

        is_fraud = random.random() < fraud_prob

        yield {"user_id": user_id, "amount": amount}, is_fraud


# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
def make_features(x):
    uid = x["user_id"]
    amount = x["amount"]

    mean = user_mean[uid].get() if user_count[uid] > 0 else amount
    var = user_var[uid].get() if user_count[uid] > 1 else 1.0

    z_score = (amount - mean) / (var + 1e-6)

    return {
        "amount": amount,
        "z_score": z_score,
        "user_mean": mean,
        "user_count": user_count[uid],
    }

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)

metric = metrics.ROCAUC()

#
for i, (x, y) in enumerate(stream_transactions()):

    uid = x["user_id"]

    features = make_features(x)

    proba = model.predict_proba_one(features).get(1, 0.0)
    metric.update(y, proba)

    model.learn_one(features, y)

    user_mean[uid].update(x["amount"])
    user_var[uid].update(x["amount"])
    user_count[uid] += 1

    if i % 500 == 0 and i > 0:
        print(f"{i} samples - ROC AUC so far: {metric.get():.4f}")

# ---------------------------
# FINAL RESULT
# ---------------------------
print("\nFINAL ROC AUC:", metric)