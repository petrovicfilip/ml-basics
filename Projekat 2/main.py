import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = seaborn.load_dataset('titanic')

df = df[["survived", "pclass", "sex", "age", "fare"]]

print(df)

df.to_csv('titanic.csv')

df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].map({'male': 0, 'female': 1})

X = df[["age", "fare", "pclass", "sex"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)

model_rf = RandomForestClassifier(n_estimators=10, random_state=42)
model_rf.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

model_svc = SVC(kernel= "linear", random_state=42)
model_svc.fit(X_train, y_train)

print("LOGISTIC REGRESSION MODEL")
y_pred = model_lr.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("RANDOM FOREST CLASSIFICATION MODEL")
y_pred = model_rf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("SVC")
y_pred = model_svc.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))