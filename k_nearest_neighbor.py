import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

def knn_classify(train_features, train_labels, test_features, k=3):
    """
    Perform K-Nearest Neighbors classification.
    
    Args:
        train_features (numpy.ndarray): Training feature vectors.
        train_labels (numpy.ndarray): Labels corresponding to the training data.
        test_features (numpy.ndarray): Test feature vectors to classify.
        k (int): Number of nearest neighbors to consider.

    Returns:
        numpy.ndarray: Predicted labels for the test data.
    """
    predictions = []
    for test_point in test_features:
        # Calculate Euclidean distance between the test point and all training points
        distances = np.linalg.norm(train_features - test_point, axis=1)
        
        # Get the indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        # Get the labels of the k nearest neighbors
        nearest_labels = train_labels[nearest_indices]
        
        # Determine the most common label (majority voting)
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    
    return np.array(predictions)

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data (replace with real image features)
    # Assuming each feature vector has been reduced to 50 dimensions
    np.random.seed(42)
    num_samples = 500
    num_classes = 5
    feature_dim = 50
    root_dir = './data/'

    # Randomly generate features and labels
    features = np.random.rand(num_samples, feature_dim)
    labels = np.random.randint(0, num_classes, num_samples)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Standardize the feature vectors (important for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Classify using KNN
    k = 5
    predictions = knn_classify(X_train, y_train, X_test, k)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"KNN Classification Accuracy: {accuracy * 100:.2f}%")
