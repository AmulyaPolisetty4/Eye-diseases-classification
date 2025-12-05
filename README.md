# Eye-diseases-classification
Deep learning-based eye disease classification system using retinal images to detect conditions like cataract, glaucoma, diabetic retinopathy, and AMD. Includes dataset preprocessing, CNN modeling, training, evaluation, and prediction.
# 🧠 Eye Diseases Classification using Deep Learning

A deep learning-based image classification system that detects multiple eye diseases from retinal fundus images.
This project uses **Convolutional Neural Networks (CNNs)** to classify retinal images into four classes:

* **Normal**
* **Diabetic Retinopathy**
* **Cataract**
* **Glaucoma**

The system includes dataset preprocessing, data augmentation, model training, evaluation, and prediction.

---

## 📌 **1. Introduction**

Eye diseases like *diabetic retinopathy, cataract,* and *glaucoma* are major causes of blindness worldwide.
Through retinal image analysis using CNNs, early disease detection becomes more accurate and automated.

This project builds a model that learns eye disease patterns from retinal images and classifies new images with high accuracy.

---

## 📁 **2. Dataset**

The dataset contains **~4000 retinal images**, collected from sources such as:

* **IDRiD**
* **HRF**
* **Ocular Recognition Dataset**
* Other public ophthalmology datasets

Each class has approximately **1000 images**:

| Class                | Description                                |
| -------------------- | ------------------------------------------ |
| Normal               | Healthy retinal images                     |
| Diabetic Retinopathy | Blood vessel leakage / abnormalities       |
| Cataract             | Clouding of the lens affecting clarity     |
| Glaucoma             | Optic nerve damage caused by high pressure |

⚠️ Dataset is **too large for GitHub**. Download link used in the project is added in the implementation.

---

## 🧼 **3. Preprocessing & Augmentation**

Steps include:

* Resizing images to 224×224
* Normalization
* Removing low-quality images
* Data Augmentation using `ImageDataGenerator`:

  * rotation
  * zoom
  * horizontal flip
  * brightness adjustments

This improves generalization and prevents overfitting.

---

## 🧠 **4. Model Architecture (CNN)**

A custom CNN model is used with:

* Convolution layers
* MaxPooling layers
* Batch Normalization
* Dropout
* Fully connected Dense layers
* Softmax activation for 4-class prediction

Optimizer: **Adam**
Loss: **Categorical Crossentropy**
Metrics: **Accuracy**

---

## 🏋️ **5. Training**

Model trained using:

```
history = model.fit(
    train_augmented,
    epochs=15,
    validation_data=valid_augmented
)
```

Training output includes:

* Accuracy (train & validation)
* Loss curves
* Epoch-wise improvements

---

## 📊 **6. Evaluation**

Metrics evaluated:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Visualization of results helps identify disease classification performance.

---

## 🔍 **7. Prediction**

Run prediction on new images:

```
model.predict(image)
```

The output shows probability for each class:

* Normal
* Diabetic Retinopathy
* Cataract
* Glaucoma

---

## 🛠️ **8. Technologies Used**

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV
* Jupyter Notebook

---

## 📦 **9. Folder Structure**

```
Eye-diseases-classification/
│── src/
│── models/
│── dataset/
│── README.md
│── requirements.txt
```

---

## 📌 **10. Future Enhancements**

* Deploy model using **Flask/Streamlit**
* Add **Grad-CAM** heatmaps for explainability
* Use transfer learning (ResNet50, EfficientNet)

---

