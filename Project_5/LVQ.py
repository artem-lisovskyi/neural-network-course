import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load a multiclass dataset
digits = load_digits()
X, y = digits.data, digits.target

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LVQ implementation with tuning
class LVQ:
    def __init__(self, prototypes, learning_rate=0.02, epochs=100):
        self.prototypes = prototypes
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                distances = np.linalg.norm(X[i] - self.prototypes, axis=1)
                winner_index = np.argmin(distances)

                if y[i] == winner_index:
                    self.prototypes[winner_index] += self.learning_rate * (X[i] - self.prototypes[winner_index])
                else:
                    self.prototypes[winner_index] -= self.learning_rate * (X[i] - self.prototypes[winner_index])
    def classify(self, X):
        predictions = []
        for i in range(len(X)):
            distances = np.linalg.norm(X[i] - self.prototypes, axis=1)
            predicted_class = np.argmin(distances)
            predictions.append(predicted_class)
        return predictions
# Tune hyperparameters
num_prototypes = len(np.unique(y_train))
learning_rate = 0.05
epochs = 200

# Initialize prototypes randomly
initial_prototypes = X_train[np.random.choice(np.where(y_train == 0)[0], num_prototypes, replace=False)]
for class_label in range(1, len(np.unique(y_train))):
    initial_prototypes = np.vstack([initial_prototypes, X_train[np.random.choice(np.where(y_train == class_label)[0], num_prototypes, replace=False)]])

# Create and train the LVQ model
lvq_model = LVQ(prototypes=initial_prototypes, learning_rate=learning_rate, epochs=epochs)
lvq_model.train(X_train, y_train)

# Make predictions on the test set
predictions = lvq_model.classify(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(f'Accuracy: {accuracy * 100:.4f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
