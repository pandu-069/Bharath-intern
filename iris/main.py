import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import tensorflow as tf
import matplotlib.pyplot as plt

# Ensure TensorFlow is installed correctly and check version
print("TensorFlow Version:", tf.__version__)

# 1. Load the iris dataset
iris = load_iris()
X = iris.data  # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target  # Target labels: Setosa, Versicolor, Virginica

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the TensorFlow Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),  # Input layer (4 input features)
    tf.keras.layers.Dense(8, activation='relu'),  # Hidden layer with 8 neurons
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons (one per class)
])

# 5. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Print the model summary
model.summary()

# 7. Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.2)

# 8. Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 9. Make predictions on the test set
y_pred = np.argmax(model.predict(X_test), axis=1)

# 10. Print classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 11. Visualize Training History
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
