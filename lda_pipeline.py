
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def run_lda_svm(X_train, X_test, y_train, y_test):
    n_classes = len(np.unique(y_train))
    lda = LDA(n_components=n_classes - 1)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    model = SVC(kernel='linear')
    model.fit(X_train_lda, y_train)

    y_pred = model.predict(X_test_lda)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm
