from scipy.stats import reciprocal, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

from ch3_mnist_demo import load_mnist

X_train, X_test, y_train, y_test = load_mnist()

svm_clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", SVC(decision_function_shape='ovr', gamma='auto'))
])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))
#
# svm_clf.fit(X_train, y_train)
#
# y_pred = svm_clf.predict(X_test)
# print(accuracy_score(y_pred, y_test))
# svm_clf = SVC(decision_function_shape="ovr", gamma="auto")
param_distributions = {"linear_svc__gamma": reciprocal(0.001, 0.1), "linear_svc__C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf_pipeline, param_distributions, n_iter=10, verbose=2, cv=3, n_jobs=-1)
rnd_search_cv.fit(X_train[:1000], y_train[:1000])
print(rnd_search_cv.best_estimator_)
print(rnd_search_cv.best_score_)
rnd_search_cv.best_estimator_.fit(X_train, y_train)
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
print(accuracy_score(y_train, y_pred))
y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))