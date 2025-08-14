# --------------------------------------------------
# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 2: Load Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Step 3: Preprocess Data (normalize pixel values to [0,1])
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Class names for output labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Step 4: Build the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 5: Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Train the Model
history = model.fit(
    x_train, y_train,
    epochs=3,
    validation_split=0.1,
    verbose=1
)

# Step 7: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Step 8: Predict on Test Set
predictions = model.predict(x_test)

# Step 9: Visualize Predictions

def plot_image(i, predictions_array, true_labels, images):
    """Display image i with predicted label and confidence.
    Uses BLUE text if correct, RED if incorrect.
    """
    pred = predictions_array[i]
    true_label = int(true_labels[i])
    img = images[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = int(np.argmax(pred))
    confidence = 100.0 * float(np.max(pred))
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(
        f"Pred: {class_names[predicted_label]} ({confidence:.2f}%)\n"
        f"True: {class_names[true_label]}",
        color=color
    )

# Show first N predictions
N = 5
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.subplot(1, N, i + 1)
    plot_image(i, predictions, y_test, x_test)

plt.tight_layout()
plt.show()
