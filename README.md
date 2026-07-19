# 🌿 Plant Disease Detection using Fine-tuned MobileNetV2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An intelligent plant disease detection system using Fine-tuned MobileNetV2 on the PlantVillage dataset**

</div>

---

## 📌 Project Overview

This project implements a **plant disease detection system** using **Deep Learning** and **Fine-tuning** techniques. By leveraging the pre-trained **MobileNetV2** model and fine-tuning it on the **PlantVillage** dataset, we have achieved a model with **90.44% accuracy** across 38 different plant disease classes.

### 🎯 Project Objectives

- Automated detection of plant diseases from leaf images
- Utilizing Fine-tuning techniques to improve accuracy with limited computational resources
- Providing a simple and user-friendly interface with Streamlit
- Optimized for resource-constrained systems (RTX 2050, 8GB RAM)

---

## ✨ Key Features

- ✅ **Detects 38 types** of plant diseases with % accuracy
- ✅ **Interactive UI** with Streamlit for image upload and diagnosis
- ✅ **Confidence level display** and top-3 probabilities for transparency
- ✅ **Optimized** for systems with 8GB RAM and mid-range GPUs
- ✅ **Comprehensive evaluation** with Precision, Recall, and Confusion Matrix
- ✅ **Error handling** and compatibility with different TensorFlow versions

---

## 🏗️ Model Architecture

### Overall Structure

```
Input (128×128×3)
    ↓
MobileNetV2 Base (Frozen - 155 layers)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (30%)
    ↓
Dense (128, ReLU)
    ↓
Dropout (20%)
    ↓
Dense (38, Softmax) ← Final Output
```

### Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Base Architecture** | MobileNetV2 (ImageNet weights) |
| **Number of Classes** | 38 plant disease classes |
| **Input Size** | 128×128 pixels (RGB) |
| **Total Parameters** | 2,426,854 |
| **Trainable Parameters** | 168,870 |
| **Frozen Parameters** | 2,257,984 |
| **Model Size** | ~9.26 MB |

### Added Layers

1. **GlobalAveragePooling2D**: Reduces MobileNetV2 output dimensions
2. **Dropout(0.3)**: Prevents Overfitting
3. **Dense(128, ReLU)**: High-level feature extraction
4. **Dropout(0.2)**: Second Dropout layer
5. **Dense(38, Softmax)**: Final classification layer

---

## 📊 Evaluation Results

### Performance on Test Dataset

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **%** | 90 out of 100 images correctly classified |
| **Precision** | **93.26%** | When the model says it's disease X, 93% of the time it's correct |
| **Recall** | **87.67%** | Out of all actual disease X cases, 87.7% were detected |

### Dataset Statistics

| Split | Number of Images | Percentage |
| :--- | :--- | :--- |
| **Train** | 37,997 | ~70% |
| **Validation** | 10,859 | ~20% |
| **Test** | 5,449 | ~10% |
| **Total** | 54,305 | 100% |

### Results Analysis

- **Precision > Recall** (93.38% vs 87.72%): The model **makes fewer false positives** (when it says a disease is present, it's usually correct), but it may **miss some disease cases** (false negatives). This is acceptable for a diagnostic system, as it's better to occasionally miss a disease than to incorrectly diagnose a healthy plant.

---

## 📁 Project Structure

```
Plant-Disease-Detection/
│
├── main.ipynb                     # Model training code
├── app.py                       # Streamlit UI application
├── requirements.txt             # Dependencies list
│
├── Dataset/                     # PlantVillage dataset
│   ├── train/                   # 37,997 images
│   ├── validation/              # 10,859 images
│   └── test/                    # 5,449 images
│
├── models/                      # Saved models
│   ├── plant_disease_model.keras
│   ├── plant_disease_model.h5
│   ├── model_architecture.json
│   └── saved_model/
│
├── outputs/                     # Generated outputs
│   ├── confusion_matrix.png
│   ├── class_names.json
│   └── model_info.json
│
└── README.md                    # Project documentation
```

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.8+
- pip
- (Optional) GPU with CUDA support

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/your-username/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download the dataset**
   - Download PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/moazeldsokyx/plantvillage)
   - Or from [GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
   - Place the `train`, `validation`, and `test` folders in the `Dataset/` directory

4. **Train the model**

```bash
python train.py
```

5. **Run the web interface**

```bash
streamlit run app.py
```

---

## 📖 Usage Guide

### Using the Streamlit Web Interface

1. **Launch the application**

```bash
streamlit run app.py
```

2. **Upload an image**
   - Click on the **Browse files** button
   - Select a leaf image (supported formats: JPG, JPEG, PNG)

3. **Diagnose the disease**
   - Click on the **🔍 Diagnose Disease** button
   - Results will display: disease name, confidence percentage, and top-3 probabilities

### Using Command Line

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model('plant_disease_model.keras')

# Preprocess the image
img = Image.open('leaf.jpg').resize((128, 128))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])
```

---

## 🧠 Technologies Used

### MobileNetV2

MobileNetV2 is a **lightweight and efficient** convolutional neural network architecture introduced by Google in 2018. Key features include:

- **Depthwise Separable Convolution**: Reduces parameters by 8-9x
- **Inverted Residuals**: Inverted residual blocks for better information flow
- **Linear Bottlenecks**: Removes non-linearities in low-dimensional layers

With **3.5 million parameters** and **71.86% accuracy** on ImageNet, it offers an excellent balance between speed and accuracy.

### Transfer Learning and Fine-tuning

**Fine-tuning** is a transfer learning method where a pre-trained model (trained on ImageNet with 14 million images) is adapted for a specific task. In this project, the base MobileNetV2 layers are frozen, and only the newly added layers (Dense, Dropout) are trained.

### Streamlit

Streamlit is an open-source framework for building data applications quickly with Python. Key features:

- **Converts Python scripts to web apps** in minutes
- **Built-in caching** for performance optimization
- **Free deployment** on Streamlit Community Cloud

---

## 📚 References

### Scientific Papers

1. **MobileNetV2 Paper**
   > Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4510–4520).

2. **PlantVillage Dataset**
   > Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics through machine learning and crowdsourcing. *arXiv preprint arXiv:1511.08060*.

3. **Mohanty et al. (2016)**
   > Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in plant science*, 7, 1419.

### Documentation

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Keras Documentation](https://keras.io/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)

### Datasets

- [PlantVillage Dataset on GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
- [PlantVillage on Kaggle](https://www.kaggle.com/datasets/moazeldsokyx/plantvillage)

---

## 📝 Citation

If you use this project in your research, please cite it as:

```bibtex
@software{plant_disease_detection_2024,
  author = {sajjadde},
  title = {Plant Disease Detection using Fine-tuned MobileNetV2},
  year = {2024},
  url = {https://github.com/your-username/Plant-Disease-Detection}
}
```

---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

---

## 👨‍💻 Developer

- **Your Name** - [GitHub](https://github.com/sajjadde)

---

## ⭐ Support

If you find this project useful, please give it a ⭐! 🌟

---

<div align="center">

**Built with ❤️ using TensorFlow and Streamlit**

</div>
