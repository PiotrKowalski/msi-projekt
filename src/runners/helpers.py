from sklearn.metrics import accuracy_score


def run_kfold(clf, filter_type):
    for fold_id, (train, test) in enumerate(clf.skf.split(clf.X, clf.y)):
        X = clf._filter_x(filter_type, train)

        clf.fit(X[train], clf.y[train])
        y_pred = clf.predict(X[test])
        clf.scores[fold_id] = accuracy_score(clf.y[test], y_pred)

    mean, std = clf.calculate_mean_and_std()

    return mean, std
