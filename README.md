# Sign Language Detection

This project implements a hand gesture recognition system using **MediaPipe**, **OpenCV**, and **Scikit-learn**. It allows users to collect gesture data, train a model to recognize ASL (A–Z) hand gestures, and run real-time inference via webcam.

---

## 📁 Project Structure

```
.
├── collect_imgs.py           # Script to collect gesture images via webcam
├── create_dataset.py         # Converts image data to landmarks and labels using MediaPipe
├── train_classifier.py       # Trains a Random Forest classifier on the extracted data
├── inference_classifier.py   # Performs real-time gesture recognition
├── requirements.txt          # Project dependencies
├── model.p                   # Trained model (generated after training)
└── data/                     # Collected dataset (generated)
```

---

## ⚙️ Setup Instructions

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Collect training data:**

   ```bash
   python collect_imgs.py
   ```

   * Press `q` to start image capture for each class.
   * Captures 100 images per class for 26 classes (A–Z by default).

3. **Generate dataset for model training:**

   ```bash
   python create_dataset.py
   ```

4. **Train the classifier:**

   ```bash
   python train_classifier.py
   ```

   * Outputs classification accuracy and saves the trained model as `model.p`.

5. **Run real-time gesture inference:**

   ```bash
   python inference_classifier.py
   ```

   * Press `q` to exit the webcam window.

---

## 🧠 Model Details

* **Features:** Normalized hand landmark coordinates (x, y)
* **Classifier:** Random Forest
* **Accuracy:** Prints test accuracy after training (typically >90% with good data)

---

## 🧾 Key Scripts Overview

### `collect_imgs.py`

* Captures 100 images for each gesture class using webcam.
* Organizes images into folders: `./data/0`, `./data/1`, ..., `./data/25`

### `create_dataset.py`

* Uses **MediaPipe Hands** to extract 21 hand landmarks from each image.
* Stores data in `data.pickle` with normalized landmark coordinates and corresponding labels.

### `train_classifier.py`

* Loads the dataset and trains a **Random Forest** model.
* Saves trained model in `model.p`.

### `inference_classifier.py`

* Loads trained model and uses webcam to recognize gestures in real-time.
* Displays predicted character with bounding box overlay.

---

## ✅ Requirements

```
opencv-python==4.7.0.68  
mediapipe==0.9.0.1  
scikit-learn==1.2.0
```

---

## 📌 Notes

* Ensure good lighting and minimal background noise during image collection for better accuracy.
* You can modify the `number_of_classes` and `dataset_size` in `collect_imgs.py` as needed.
* For better performance, consider using more advanced classifiers or deep learning models (e.g., CNN with TensorFlow or PyTorch).


