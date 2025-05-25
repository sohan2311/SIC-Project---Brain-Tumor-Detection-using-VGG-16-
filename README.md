# SIC-Project---Brain-Tumor-Detection-using-VGG-16

# 🧠 Brain Tumor Detection Using VGG-16 Model

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/username/brain-tumor-detection?style=social)](https://github.com/username/brain-tumor-detection/stargazers)

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using the **VGG-16 architecture** for detecting brain tumors from MRI scans. The model achieves **~88% training accuracy** and **~82% test accuracy** using transfer learning techniques.

### 👨‍🎓 Project Information
- **Author**: Sohan Maity
- **Roll No**: 523EC0001
- **Email**: 523ec0001@iiitk.ac.in
- **Institution**: Indian Institute of Information Technology Design and Manufacturing Kurnool (IIITDM Kurnool)
- **Program**: Samsung Innovation Campus - Final Project

## 🎥 Project Presentation

[![Project Video](https://img.shields.io/badge/Watch%20Project%20Video-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/V4Sew-QPm9o?si=hvokTJpky2-1sNlq)

## 📊 Dataset Information

[![Download Dataset](https://img.shields.io/badge/Download%20Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset contains MRI brain scan images classified into two categories:
- **NO**: No tumor detected (encoded as 0)
- **YES**: Tumor detected (encoded as 1)

### Dataset Statistics
| Set | Images | Accuracy |
|-----|--------|----------|
| Training Set | 169 images | ~88% |
| Test Set | 84 images | ~82% |

## 🏗️ Model Architecture

### VGG-16 Transfer Learning
- **Base Model**: VGG-16 pre-trained on ImageNet
- **Input Shape**: (224, 224, 3)
- **Frozen Layers**: All VGG-16 convolutional layers
- **Custom Head**: Global Average Pooling + Dense layers

### Model Structure
```
VGG-16 Base (Frozen)
    ↓
Global Average Pooling 2D
    ↓
Dense(1024, activation='relu')
    ↓
Dense(1024, activation='relu')
    ↓
Dense(512, activation='relu')
    ↓
Dense(2, activation='softmax')
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow keras opencv-python matplotlib scikit-learn plotly tqdm imutils numpy
```

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
   ```

### Usage
1. **Train the model**
   ```python
   python train_model.py
   ```

2. **Make predictions**
   ```python
   python predict.py --image path/to/mri_scan.jpg
   ```

## 📁 Project Structure
```
brain-tumor-detection/
│
├── data/
│   ├── brain_tumor_dataset/
│   │   ├── yes/          # Tumor images
│   │   └── no/           # No tumor images
│   ├── TRAIN/
│   ├── TEST/
│   └── VAL/
│
├── models/
│   └── brain_tumor_model.h5
│
├── notebooks/
│   └── Brain_Tumor_Detection.ipynb
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── data_preprocessing.py
│   └── model_architecture.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔬 Technical Details

### Data Preprocessing
- **Image Resizing**: 224×224 pixels
- **Normalization**: Pixel values scaled to [0, 1]
- **Label Encoding**: Binary classification (0: No Tumor, 1: Tumor)
- **Data Split**: 67% Training, 33% Testing

### Model Configuration
- **Optimizer**: Adam (learning_rate=0.01)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 15
- **Batch Size**: Default

### Performance Metrics
| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 88% | 82% |
| Loss | 0.0007 | 0.6297 |

## 📈 Results & Visualizations

### Training History
The model shows excellent learning progression:
- **Epoch 1**: 62.47% → **Epoch 15**: 100% (Training Accuracy)
- **Validation Accuracy**: Stabilized around 89-90%

### Key Findings
- ✅ Successful implementation of transfer learning
- ✅ Good generalization despite small dataset
- ✅ Effective feature extraction using VGG-16
- ⚠️ Signs of overfitting in later epochs

## 🧠 About Brain Tumors

A brain tumor is an abnormal mass of cells in the brain, which can be:
- **Benign**: Non-cancerous
- **Malignant**: Cancerous

**Common Symptoms:**
- Persistent headaches
- Seizures
- Vision/speech difficulties
- Memory loss
- Personality changes

**Early detection through MRI analysis can significantly improve treatment outcomes.**

## 🛠️ Technologies Used

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## 📋 Requirements

```
tensorflow>=2.0.0
keras>=2.0.0
opencv-python>=4.0.0
matplotlib>=3.0.0
scikit-learn>=1.0.0
plotly>=5.0.0
tqdm>=4.0.0
imutils>=0.5.0
numpy>=1.19.0
```

## 🚀 Future Enhancements

- [ ] **Improve Accuracy**: Implement ResNet, EfficientNet architectures
- [ ] **Data Augmentation**: Add rotation, zoom, flipping techniques
- [ ] **Multi-class Classification**: Detect different tumor types (Glioma, Meningioma, Pituitary)
- [ ] **Web Deployment**: Create Flask/FastAPI web application
- [ ] **Mobile App**: Develop mobile application for real-time detection
- [ ] **Model Optimization**: Implement model quantization for faster inference

## 📊 Model Performance Visualization

The training process shows:
- **Rapid Learning**: Quick improvement in first 5 epochs
- **Convergence**: Model stabilizes around epoch 7-8
- **Overfitting Signs**: Training accuracy reaches 100% while validation plateaus

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Samsung Innovation Campus** for providing the learning platform
- **IIITDM Kurnool** for academic support
- **Kaggle Community** for the dataset
- **TensorFlow/Keras Teams** for the excellent deep learning frameworks
- **VGG Team** for the groundbreaking architecture

## 📞 Contact

**Sohan Maity**
- 📧 Email: 523ec0001@iiitk.ac.in
- 🎓 Institution: IIITDM Kurnool
- 📱 Roll No: 523EC0001

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/username)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/username)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

*Made with ❤️ for advancing medical AI and brain tumor detection*
