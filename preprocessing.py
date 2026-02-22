
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_and_split(X_features, y, test_size=0.2, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
