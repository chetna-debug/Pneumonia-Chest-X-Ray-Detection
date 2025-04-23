import os
import numpy as np
import streamlit as st
from PIL import Image
import h5py

# ============================
# 🔹 Define Sigmoid Activation
# ============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative for backpropagation

# ================================
# 🔹 Initialize Model Parameters
# ================================
input_size = 1024  # Assume we resize images to 32x32 (flattened to 1024 pixels)
W = np.random.randn(1, input_size) * 0.01  # Small random weights
b = np.zeros((1, 1))  # Bias initialized to zero

# ================================
# 🔹 Check for Existing Weights
# ================================
WEIGHTS_FILE = "./model/slp_weights.h5"

if os.path.exists(WEIGHTS_FILE):
    st.success("Loading existing model weights...")
    with h5py.File(WEIGHTS_FILE, "r") as f:
        W = np.array(f["W"])
        b = np.array(f["b"])
else:
    st.warning("No pretrained weights found. Model will start with random weights.")

# ============================
# 🔹 Forward Propagation
# ============================
def forward_propagation(X):
    z = np.dot(W, X) + b  # Linear transformation
    return sigmoid(z)  # Apply activation function

# ============================
# 🔹 Training Function
# ============================
def train(X_train, y_train, epochs=1000, learning_rate=0.01):
    global W, b
    m = X_train.shape[1]  # Number of training samples

    for epoch in range(epochs):
        # Forward pass
        y_pred = forward_propagation(X_train)

        # Compute Loss (Binary Cross-Entropy)
        loss = -(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)).mean()

        # Backpropagation
        dz = y_pred - y_train  # Error
        dW = np.dot(dz, X_train.T) / m  # Gradient w.r.t. weights
        db = np.sum(dz) / m  # Gradient w.r.t. bias

        # Update Weights
        W -= learning_rate * dW
        b -= learning_rate * db

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            st.write(f"Epoch {epoch} | Loss: {loss:.4f}")

    # Save Trained Weights
    with h5py.File(WEIGHTS_FILE, "w") as f:
        f.create_dataset("W", data=W)
        f.create_dataset("b", data=b)
    st.success("Model training complete and weights saved!")

# ==============================
# 🔹 Streamlit UI
# ==============================
st.title("Pneumonia Classification with SLP")
st.header("Upload a Chest X-ray Image")

file = st.file_uploader("Upload an image", type=['jpeg', 'jpg', 'png'])

if file:
    image = Image.open(file).convert("L")  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to match input size
    image_array = np.array(image).flatten().reshape(-1, 1) / 255.0  # Normalize

    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Make Prediction
    prediction = forward_propagation(image_array)

    if prediction >= 0.5:
        st.error("Prediction: Pneumonia Detected 🚨")
    else:
        st.success("Prediction: No Pneumonia ✅")

