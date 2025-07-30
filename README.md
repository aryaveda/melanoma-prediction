# Melanoma Prediction System

Sistem prediksi melanoma berbasis AI yang menggunakan model hybrid CNN-ViT untuk mendiagnosis berbagai jenis lesi kulit dengan akurasi tinggi.

## 🎯 Fitur Utama

- **Prediksi Multi-Kelas**: Mendiagnosis 9 jenis lesi kulit berbeda:
  - AK (Actinic Keratosis)
  - BCC (Basal Cell Carcinoma)
  - BKL (Benign Keratosis-like Lesions)
  - DF (Dermatofibroma)
  - SCC (Squamous Cell Carcinoma)
  - VASC (Vascular Lesions)
  - Melanoma
  - Nevus
  - Unknown

- **Model Hybrid**: Menggabungkan CNN (EfficientNet) dan Vision Transformer (ViT) untuk performa optimal
- **Metadata Integration**: Menggunakan informasi usia dan jenis kelamin untuk meningkatkan akurasi prediksi
- **Grad-CAM Visualization**: Menampilkan area yang menjadi fokus model dalam membuat prediksi
- **Ground Truth Matching**: Sistem pencocokan dengan data ground truth untuk validasi
- **Web Interface**: Antarmuka web yang user-friendly untuk upload gambar dan melihat hasil

## 🚀 Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch, Timm, Transformers
- **Computer Vision**: OpenCV, Albumentations
- **Model Architecture**: EfficientNet-B5 + Vision Transformer
- **Visualization**: Grad-CAM
- **Deployment**: Vercel, Docker

## 📋 Prerequisites

- Python 3.8+
- CUDA (opsional, untuk GPU acceleration)
- Git

## 🛠️ Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd melanomaPredict
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model
Model akan otomatis di-download saat pertama kali menjalankan aplikasi, atau Anda dapat mengunduh manual:
```bash
# Model akan di-download ke folder models/
# File: effnetb5_384_9c_50epo_ext_BEST_epoch48.pth
```

## 🏃‍♂️ Cara Menjalankan

### Development Mode
```bash
python app.py
```
Aplikasi akan berjalan di `http://localhost:5000`

### Production Mode
```bash
gunicorn app:app
```

### Docker
```bash
docker build -t melanoma-predict .
docker run -p 5000:5000 melanoma-predict
```

## 📖 Cara Penggunaan

1. **Buka Aplikasi**: Akses `http://localhost:5000`
2. **Upload Gambar**: Pilih file gambar lesi kulit (format: JPG, PNG)
3. **Input Metadata**: Masukkan usia dan pilih jenis kelamin
4. **Prediksi**: Klik tombol "Predict" untuk mendapatkan hasil
5. **Analisis**: Lihat hasil prediksi, confidence score, dan visualisasi Grad-CAM

## 🏗️ Arsitektur Sistem

### Model Architecture
- **Backbone**: EfficientNet-B5 (CNN)
- **Transformer**: Vision Transformer (ViT)
- **Fusion**: Attention mechanism untuk menggabungkan fitur CNN dan ViT
- **Metadata**: Attention layer untuk mengintegrasikan informasi usia dan jenis kelamin

### Data Processing
- **Image Preprocessing**: Resize, normalization, augmentation
- **Metadata Processing**: Age normalization, sex encoding
- **Ground Truth Matching**: Fuzzy matching dengan dataset ground truth

### Web Interface
- **Home**: Landing page dengan informasi sistem
- **Predictor**: Halaman utama untuk prediksi
- **About**: Informasi tentang proyek dan tim
- **Skin Info**: Panduan tentang jenis-jenis lesi kulit

## 📊 Performa Model

- **Accuracy**: Tinggi pada dataset training
- **Classes**: 9 kelas lesi kulit
- **Input Size**: 384x384 pixels
- **Augmentation**: Albumentations untuk training robustness

## 🔧 Konfigurasi

### Environment Variables
```bash
# Opsional: Set untuk production
FLASK_ENV=production
FLASK_DEBUG=0
```

### Model Configuration
- **Image Size**: 384x384
- **Batch Size**: Sesuai dengan GPU memory
- **Learning Rate**: Optimized untuk training
- **Temperature Scaling**: 1.0 (untuk calibration)

## 📁 Struktur Proyek

```
melanomaPredict/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── dockerfile            # Docker configuration
├── vercel.json          # Vercel deployment config
├── models/              # Model weights directory
├── templates/           # HTML templates
│   ├── index.html
│   ├── predict.html
│   ├── about.html
│   └── skin_info.html
├── static/              # Static files (CSS, JS, images)
├── groundtruth.csv      # Ground truth dataset
└── README.md           # This file
```

## 🚀 Deployment

### Vercel
Proyek sudah dikonfigurasi untuk deployment di Vercel:
- `vercel.json`: Konfigurasi deployment
- `runtime.txt`: Python version specification
- `.vercelignore`: File yang di-exclude

### Docker
```bash
# Build image
docker build -t melanoma-predict .

# Run container
docker run -p 5000:5000 melanoma-predict
```

## 🤝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 License

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## ⚠️ Disclaimer

Sistem ini dibuat untuk tujuan penelitian dan pendidikan. **TIDAK** dimaksudkan untuk menggantikan diagnosis medis profesional. Selalu konsultasikan dengan dokter untuk diagnosis yang akurat.

## 📞 Support

Jika Anda memiliki pertanyaan atau masalah:
- Buat issue di repository
- Hubungi tim pengembang
- Lihat dokumentasi di folder `docs/`

---

**Dibuat dengan ❤️ untuk membantu diagnosis lesi kulit yang lebih akurat** 