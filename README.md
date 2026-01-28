# ASL Sign Language Detector 🤟

একটি Machine Learning based ASL (American Sign Language) sign detection application যা Streamlit দিয়ে তৈরি।

## 📁 Project Structure

```
ASL/
├── dataset/               # ASL images dataset
│   ├── A-samples/
│   ├── B-samples/
│   └── ... (Y-samples পর্যন্ত)
├── train_model.py        # Model training script
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📚 Usage

### Step 1: Train the Model

প্রথমে আপনাকে model train করতে হবে:

```bash
python train_model.py
```

এটি:
- Dataset থেকে সব images load করবে
- CNN model তৈরি করবে
- Model train করবে (প্রায় 50 epochs)
- Best model save করবে (`asl_model_best.h5`)
- Training history plot save করবে (`training_history.png`)

**Note:** Training এ সময় লাগবে। GPU থাকলে দ্রুত হবে।

### Step 2: Run the Streamlit App

Model training শেষ হলে app run করুন:

```bash
streamlit run app.py
```

Browser এ app খুলবে (সাধারণত `http://localhost:8501`)

## 🎯 Features

- **Image Upload:** ASL sign এর ছবি upload করুন
- **Real-time Prediction:** Instant prediction পাবেন
- **Confidence Score:** কতটা accurate prediction তা দেখুন
- **Top 5 Predictions:** সবচেয়ে সম্ভাব্য 5টি prediction
- **Beautiful UI:** User-friendly Streamlit interface

## 🏗️ Model Architecture

- **Type:** Convolutional Neural Network (CNN)
- **Input Size:** 128x128x3 (RGB images)
- **Layers:**
  - 4 Convolutional blocks (32, 64, 128, 256 filters)
  - BatchNormalization এবং Dropout layers
  - 2 Dense layers (512, 256 units)
  - Softmax output layer (24 classes)
- **Optimizer:** Adam
- **Data Augmentation:** Rotation, shift, flip, zoom

## 📊 Supported Signs

Model নিচের ASL letters detect করতে পারে:
**A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y**

(J এবং Z motion require করে, তাই included নয়)

## 💡 Tips for Best Results

- Clear, well-lit ছবি use করুন
- Hand clearly visible হতে হবে
- Simple background রাখুন
- Focused ছবি use করুন

## 🛠️ Troubleshooting

### Model file not found error
এটা দেখালে প্রথমে `python train_model.py` run করুন

### Low accuracy
- More epochs train করুন
- Data augmentation adjust করুন
- Model architecture modify করুন

### App না খুললে
Port busy থাকলে:
```bash
streamlit run app.py --server.port 8502
```

## 📝 Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Streamlit 1.29.0
- OpenCV 4.8.1
- NumPy, Matplotlib, scikit-learn

## 🎓 Model Training Details

- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Train/Test Split:** 80/20
- **Callbacks:**
  - ModelCheckpoint: Best model save করে
  - EarlyStopping: Overfitting prevent করে
  - ReduceLROnPlateau: Learning rate adjust করে

## 📈 Next Steps

1. Model train করুন: `python train_model.py`
2. App run করুন: `streamlit run app.py`
3. ASL sign এর ছবি upload করে test করুন!

## ❤️ Made With

- TensorFlow/Keras - Deep Learning
- Streamlit - Web Interface
- OpenCV - Image Processing
- Python - Programming Language

---

**Happy Learning! 🤟**
"# CGIP_project_4_1" 
