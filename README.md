# 🔢 Handwritten Digit Recognition using CNN

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">

<img src="https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">

<img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" alt="Streamlit">

<img src="https://img.shields.io/badge/CNN-Deep%20Learning-purple?style=for-the-badge" alt="CNN">

<img src="https://img.shields.io/badge/MNIST-Dataset-green?style=for-the-badge" alt="MNIST">

</p>

<h3 align="center">
🧠 An End-to-End Deep Learning Application for Real-Time Handwritten Digit Recognition
</h3>

<p align="center">
Draw a digit ✍️ → CNN processes it → Get an instant prediction 🎯
</p>

---

## 🚀 Live Demo

<p align="center">

### ✨ Try the application

**👉 (https://mnist-handwritten-digit-recognition-gayatripagare.streamlit.app/)**

</p>



---

## 📌 Project Overview

This project is an **end-to-end handwritten digit recognition system** built using a **Convolutional Neural Network (CNN)** and deployed as an interactive **Streamlit web application**.

The model is trained on the **MNIST handwritten digit dataset**, which contains grayscale images of digits from **0 to 9**.

The application allows users to:

* ✍️ Draw a handwritten digit directly on the screen
* 🔮 Get a real-time prediction
* 📊 View prediction confidence
* 📈 View probability distribution for all 10 digits
* 📷 Upload an image for prediction *(if enabled in the application)*

---

## 🎯 Project Goal

The goal of this project is to demonstrate the complete Deep Learning workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
CNN Architecture
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Integration
   ↓
Real-Time Prediction
```

---

## 🧠 How It Works

```text
                ┌──────────────────┐
                │   User Drawing   │
                │       ✍️         │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Image Processing │
                │ Grayscale        │
                │ Resize 28×28     │
                │ Normalize        │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │   CNN Model      │
                │ TensorFlow/Keras │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │   Prediction     │
                │      0 - 9       │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │   Confidence     │
                │      📊          │
                └──────────────────┘
```

---

## 📊 Dataset

### MNIST

The model uses the MNIST handwritten digit dataset.

| Property        |         Value |
| --------------- | ------------: |
| Training Images |        60,000 |
| Testing Images  |        10,000 |
| Image Size      |       28 × 28 |
| Channels        | 1 (Grayscale) |
| Classes         |            10 |
| Classes         |         0 – 9 |

Each image contains **784 pixels (28 × 28)**.

---

## 🧹 Data Preprocessing

The images go through the following preprocessing pipeline:

```text
Raw Pixel Values
       ↓
Convert to Float32
       ↓
Normalize / 255
       ↓
Reshape
       ↓
28 × 28 × 1
       ↓
CNN Input
```

Pixel values are normalized from:

```text
0 – 255
```

to:

```text
0 – 1
```

This helps the neural network train more effectively.

---

## 🏗️ CNN Architecture

The model uses multiple convolutional layers to learn visual features from handwritten digits.

```text
Input
28 × 28 × 1
     │
     ▼
┌────────────────────┐
│ Conv2D - 32 Filters│
│ ReLU               │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Max Pooling 2×2    │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Conv2D - 64 Filters│
│ ReLU               │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Max Pooling 2×2    │
└─────────┬──────────┘
          ▼
┌─────────────────────┐
│ Conv2D - 128 Filters│
│ ReLU                │
└──────────┬──────────┘
           ▼
       Flatten
           │
           ▼
      Dense(128)
           │
           ▼
       Dropout
           │
           ▼
     Dense(10)
           │
           ▼
       Softmax
           │
           ▼
      Digit 0 – 9
```

---

## 🔥 Model Improvements

The improved model uses techniques designed to improve generalization and robustness.

### Data Augmentation

The training images are slightly modified using:

* 🔄 Rotation
* ↔️ Width shifting
* ↕️ Height shifting
* 🔍 Zoom

This helps the model handle handwritten digits that may be positioned or written differently.

### Early Stopping

Training automatically stops when validation performance stops improving, while restoring the best model weights.

### Learning Rate Reduction

The learning rate is reduced when validation loss stops improving.

---

## 📈 Model Performance

### Baseline CNN

Your initial CNN achieved:

> **99.14% test accuracy**

The improved model was then trained using augmentation and callbacks.

### Final Model

| Metric            |                   Result |
| ----------------- | -----------------------: |
| Test Accuracy     | **YOUR_FINAL_ACCURACY%** |
| Input Size        |              28 × 28 × 1 |
| Number of Classes |                       10 |
| Optimizer         |                     Adam |
| Loss Function     | Categorical Crossentropy |

> 📝 Replace `YOUR_FINAL_ACCURACY%` with the actual accuracy from your final training run.

---

## 🆚 Why CNN?

A CNN is particularly suitable for image classification because it can automatically learn spatial features such as:

```text
Edges
  ↓
Curves
  ↓
Shapes
  ↓
Digit Patterns
  ↓
Digit Classification
```

Instead of manually creating image features, the CNN learns useful features during training.

---

## 🎨 Streamlit Application

The trained model is integrated into an interactive Streamlit application.

### User Flow

```text
✍️ Draw Digit
      ↓
🖼️ Capture Image
      ↓
⚙️ Preprocess Image
      ↓
🧠 CNN Prediction
      ↓
🎯 Predicted Digit
      ↓
📊 Confidence Score
```

Example:

```text
┌──────────────────────────────┐
│      Draw a Digit ✍️         │
│                              │
│             7                │
│                              │
└──────────────────────────────┘

        🔮 Predict Digit

       🎯 Prediction: 7

       📊 Confidence: 99.xx%
```

---

## 🛠️ Technologies Used

| Technology    | Purpose              |
| ------------- | -------------------- |
| 🐍 Python     | Programming          |
| 🧠 TensorFlow | Deep Learning        |
| 🔥 Keras      | CNN development      |
| 🖼️ NumPy     | Numerical processing |
| 🐼 Pandas     | Data handling        |
| 📊 Matplotlib | Visualization        |
| 🎨 Streamlit  | Web application      |
| 🐙 GitHub     | Version control      |

---

## 📁 Project Structure

```text
MNIST-Handwritten-Digit-Recognition/
│
├── app.py
│
├── mnist_cnn_improved.keras
│
├── CNN-Mnist.ipynb
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2️⃣ Navigate to the project

```bash
cd MNIST-Handwritten-Digit-Recognition
```

### 3️⃣ Create a virtual environment

```bash
python -m venv venv
```

### 4️⃣ Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### 5️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Model File

The trained model is saved as:

```text
mnist_cnn_improved.keras
```

The Streamlit application loads this trained model and uses it to make predictions.

---

## 💡 Key Learning Outcomes

Through this project, I learned and implemented:

* ✅ Image preprocessing
* ✅ CNN architecture design
* ✅ Convolution and pooling
* ✅ Model training
* ✅ Data augmentation
* ✅ Early stopping
* ✅ Learning-rate scheduling
* ✅ Model evaluation
* ✅ Model saving/loading
* ✅ Image preprocessing for inference
* ✅ Streamlit application development
* ✅ Real-time Deep Learning prediction
* ✅ GitHub project organization

---

## 🔮 Future Improvements

Possible future improvements include:

* [ ] 📷 Upload handwritten digit images
* [ ] 🎨 Improve drawing canvas
* [ ] 🧠 Add model confidence visualization
* [ ] 📊 Add confusion matrix
* [ ] 🔍 Add Grad-CAM visualization
* [ ] 📱 Improve mobile responsiveness
* [ ] 🚀 Deploy the application publicly
* [ ] 🌐 Add prediction history
* [ ] 📈 Add model performance dashboard

---

## 🧪 Example Prediction

```text
Input:
    ✍️ Handwritten Digit

        ↓

CNN Model

        ↓

Prediction:
    🎯 7

Confidence:
    📊 99.xx%
```

---

## 📚 Project Files

| File                                                     | Description                           |
| -------------------------------------------------------- | ------------------------------------- |
| [`app.py`](./app.py)                                     | Streamlit application                 |
| [`CNN-Mnist.ipynb`](./CNN-Mnist.ipynb)                   | Model development and experimentation |
| [`mnist_cnn_improved.keras`](./mnist_cnn_improved.keras) | Trained CNN model                     |
| [`requirements.txt`](./requirements.txt)                 | Python dependencies                   |

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

If you have an idea that could improve the project:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Open a Pull Request

---

## ⭐ Support

If you found this project interesting, consider giving it a ⭐ on GitHub!

---

## 👩‍💻 Author

### Gayatri Pagare

**AI & Data Science Student | Python | Machine Learning | Deep Learning**

I'm currently building projects in:

```text
Python
Machine Learning
Deep Learning
SQL
Data Visualization
Streamlit
```

---

<p align="center">

### 🧠 Built with Deep Learning + Python + Streamlit

**From handwritten pixels → intelligent prediction 🚀**

</p>
