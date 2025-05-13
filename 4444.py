
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.svm import SVC



plt.figure(figsize=(13, 17))
plt.suptitle("Сравнение методов классификации sklearn", fontsize=10, y=1.2)

classifiers = [

    ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    (" GradientBoosting",
     GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42))
    ,
    ("SVС", SVC(kernel='rbf', C=1.0))

]

datasets = [
    ("Сircles", datasets.make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=30)),
    ("Moons", datasets.make_moons(n_samples=500, noise=0.05, random_state=30)),
    ("gaussianquantile", datasets.make_gaussian_quantiles(n_features=2, n_classes=3, random_state=0)),
    ("different blobs", datasets.make_blobs(n_samples=500, cluster_std=[1.0, 0.5], random_state=30, centers=2)),
   ("Intersecting blobs", datasets.make_blobs(n_samples=500, random_state=30, centers=2))
]

for dataset_idx, (dataset_name, (X, y)) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    for clf_idx, (clf_name, clf) in enumerate(classifiers):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        y_pred = clf.predict(X_test)

        plt.subplot(len(datasets), len(classifiers), dataset_idx * len(classifiers) + clf_idx + 1)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        for i in range(len(X_train)):
            plt.scatter(X_train[i, 0], X_train[i, 1], marker='o' if y_train[i] else 'x',
                        c='blue', alpha=0.5, s=20)

        for i in range(len(X_test)):
            marker = 'o' if y_test[i] else 'x'
            color = 'green' if y_test[i] == y_pred[i] else 'pink'
            if marker == 'o':
                plt.scatter(X_test[i, 0], X_test[i, 1], marker=marker, c=color, edgecolors='k', s=30)
            else:
                plt.scatter(X_test[i, 0], X_test[i, 1], marker=marker, c=color, s=30)

        if dataset_idx == 0:
            plt.title(clf_name)
        if clf_idx == 0:
            plt.ylabel(dataset_name)

        plt.xticks(())
        plt.yticks(())

plt.tight_layout()
plt.show()