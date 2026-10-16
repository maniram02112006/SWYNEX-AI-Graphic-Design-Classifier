
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Dataset location
data_dir = (
    "/content/graphic_design_classifier_dataset/"
    "graphic_design_classifier_dataset"
)

categories = [
    "cinematic",
    "minimalist",
    "typography",
    "sports",
    "vintage"
]

# 2. Load images and labels
images = []
labels = []

for label, category in enumerate(categories):
    folder = os.path.join(data_dir, category)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Missing category folder: {folder}"
        )

    for filename in os.listdir(folder):
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            path = os.path.join(folder, filename)

            image = keras.utils.load_img(
                path, target_size=(160, 160)
            )
            image = keras.utils.img_to_array(image)

            images.append(image)
            labels.append(label)

X = np.array(images, dtype="float32")
y = np.array(labels, dtype="int32")

print("Total images:", len(X))
print("Categories:", categories)

# Check the dataset
if len(X) < 10 or len(np.unique(y)) != 5:
    raise ValueError(
        "Please check that all five folders contain images."
    )

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

# 4. Load a pretrained image feature extractor
base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# 5. Build the classification model
inputs = keras.Input(shape=(160, 160, 3))

x = keras.applications.mobilenet_v2.preprocess_input(
    inputs
)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(categories),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# 6. Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=4,
    verbose=1
)

# 8. Evaluate the model
loss, accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

print("\nTest loss:", round(float(loss), 4))
print("Test accuracy:", round(float(accuracy), 4))

# 9. Display classification metrics
predictions = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predicted_labels,
    labels=list(range(len(categories))),
    target_names=categories,
    zero_division=0
))

# 10. Save the trained model
model.save(
    "/content/graphic_design_classifier.keras"
)

print("\nTraining completed successfully!")
print("Model saved as graphic_design_classifier.keras")
