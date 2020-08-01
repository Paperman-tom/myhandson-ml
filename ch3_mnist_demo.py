import numpy as np
from sklearn.datasets import fetch_openml, fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]
    return mnist


def fetch_mnist():
    try:
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
        return sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
    except ImportError:
        mnist = fetch_mldata('MNIST original')
        return mnist


def load_mnist():
    """
    取mnist数据集，并进行数据洗牌
    :return:X_train训练集数据, X_test测试集数据, y_train训练集标签, y_test测试集标签
    """
    mnist = fetch_mnist()
    X, y = mnist['data'], mnist['target']
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_mnist()
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train)
    y_pred = sgd_clf.predict(X_test)
    print("sgd_clf acc: " + str(accuracy_score(y_pred, y_test)))
    param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    y_pred = knn_clf.predict(X_test)
    print("knn_clf acc: " + str(accuracy_score(y_pred, y_test)))
