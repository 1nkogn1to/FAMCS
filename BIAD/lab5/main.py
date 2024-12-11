import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import expit

np.random.seed(1337)

def set_parameters():
    mu_pos = np.array([2, 2])
    mu_neg = np.array([0, 0])
    sigma = np.array([[1, 0.8], [0.8, 1]])
    return mu_pos, mu_neg, sigma

def generate_data(mu_pos, mu_neg, sigma):
    N = 30
    X_pos = np.random.multivariate_normal(mu_pos, sigma, N)
    X_neg = np.random.multivariate_normal(mu_neg, sigma, N)
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(N), np.zeros(N)))
    return X, y

def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def plot_data(X_train, y_train, X_val, y_val, X_test, y_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, label='Training Data', cmap='bwr')
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, label='Validation Data', cmap='coolwarm', marker='x')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label='Test Data', cmap='winter', marker='v')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def sigmoid(z):
    return expit(z)

def train_logistic_regression(X_train, y_train, learning_rate=0.01, epochs=1000):
    m, n = X_train.shape
    w = np.zeros(n)
    b = 0
    
    for _ in range(epochs):
        z = np.dot(X_train, w) + b
        a = sigmoid(z)
        dz = a - y_train
        dw = np.dot(X_train.T, dz) / m
        db = np.sum(dz) / m
        w -= learning_rate * dw
        b -= learning_rate * db
    return w, b

def predict_proba(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def plot_decision_boundary(X, y, w, b):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    
    x_values = [np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)]
    y_values = - (b + np.dot(w[0], x_values)) / w[1]
    
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def print_probabilities(X, y, w, b):
    probabilities = predict_proba(X, w, b)
    for i in range(5):
        print(f"Object: {X[i]}, True Label: {y[i]}, Predicted Probability: {probabilities[i]:.4f}")

def find_optimal_threshold(X_val, y_val, w, b):
    probabilities = predict_proba(X_val, w, b)
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_recall = 0
    
    for t in thresholds[::-1]:
        preds = (probabilities >= t).astype(int)
        recall = my_recall_score(y_val, preds)
        if recall >= 0.6:
            best_threshold = t
            best_recall = recall
            break
    
    print(f"Optimal Threshold: {best_threshold:.2f}, Recall: {best_recall:.2f}")
    return best_threshold

def my_accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def my_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def my_precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def my_recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def my_roc_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr = []
    fpr = []
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    
    return np.array(fpr), np.array(tpr), thresholds

def my_auc(fpr, tpr):
    return np.trapz(tpr, fpr)

def evaluate_model(X_test, y_test, w, b, t):
    probabilities = predict_proba(X_test, w, b)
    preds = (probabilities >= t).astype(int)
    
    acc = my_accuracy_score(y_test, preds)
    cm = my_confusion_matrix(y_test, preds)
    precision = my_precision_score(y_test, preds)
    recall = my_recall_score(y_test, preds)
    
    fpr, tpr, _ = my_roc_curve(y_test, probabilities)
    roc_auc = my_auc(fpr, tpr)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def main():
    mu_pos, mu_neg, sigma = set_parameters()
    X, y = generate_data(mu_pos, mu_neg, sigma)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    
    plot_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    w, b = train_logistic_regression(X_train, y_train)
    plot_decision_boundary(X, y, w, b)
    
    print_probabilities(X, y, w, b)
    
    t = find_optimal_threshold(X_val, y_val, w, b)
    evaluate_model(X_test, y_test, w, b, t)
    
if __name__ == "__main__":
    main()