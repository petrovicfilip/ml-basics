from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

data = iris.data
target = iris.target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.2, random_state = 42)

model = KNeighborsClassifier(n_neighbors = 3)
log_model = LogisticRegression(max_iter=200)

model.fit(data_train, target_train)
log_model.fit(data_train, target_train)

predictions = model.predict(data_test)
predictions_log = log_model.predict(data_test)

print("\n KNN Classifier")
print(classification_report(target_test, predictions))
print("Accuracy: ", accuracy_score(target_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(target_test, predictions))

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(target_test, predictions_log))
print("Confusion matrix:")
print(confusion_matrix(target_test, predictions_log))
print(classification_report(target_test, predictions_log))