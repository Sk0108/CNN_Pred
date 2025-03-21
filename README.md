# CNN_Pred
# ACM SIGKDD R&amp;D Task 
# 🧠 CNN Digit Classifier on MNIST Dataset

A Convolutional Neural Network (CNN) model built using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The trained model is saved as an `.h5` file and visualized using Netron.

---

## 📊 Dataset

- **Dataset:** [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
- **Classes:** 10 (Digits 0-9)
- **Image Size:** 28x28 pixels, grayscale

---

## 🚀 Model Architecture

| Layer            | Filters/Units | Kernel Size | Activation | Extra                  |
|------------------|---------------|-------------|------------|------------------------|
| Input Layer      | -             | -           | -          | (28x28x1)              |
| Conv2D           | 32            | 3x3         | ReLU       | BatchNorm + MaxPooling |
| Conv2D           | 64            | 3x3         | ReLU       | BatchNorm + MaxPooling |
| Conv2D           | 128           | 3x3         | ReLU       | BatchNorm + MaxPooling |
| Flatten          | -             | -           | -          | -                      |
| Dense            | 128           | -           | ReLU       | BatchNorm + Dropout    |
| Dense (Output)   | 10            | -           | Softmax    | -                      |

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mnist-cnn.git
   cd mnist-cnn
