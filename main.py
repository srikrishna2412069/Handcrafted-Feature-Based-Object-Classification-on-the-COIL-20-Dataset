
from dataset_loader import load_dataset
from hog_extraction import extract_hog_features
from preprocessing import scale_and_split
from pca_pipeline import run_pca_svm
from lda_pipeline import run_lda_svm

DATASET_PATH = "dataset"  # Update this path

def main():
    X, y = load_dataset(DATASET_PATH)
    X_features = extract_hog_features(X)
    X_train, X_test, y_train, y_test = scale_and_split(X_features, y)

    acc_pca, cm_pca = run_pca_svm(X_train, X_test, y_train, y_test)
    print("PCA + SVM Accuracy:", acc_pca)

    acc_lda, cm_lda = run_lda_svm(X_train, X_test, y_train, y_test)
    print("LDA + SVM Accuracy:", acc_lda)

if __name__ == "__main__":
    main()
