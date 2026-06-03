# 🚘 Driver Monitoring System
**AI-Powered Drowsiness & Distraction Detection**

Gerçek zamanlı yapay zeka destekli sürücü izleme sistemi. Uzun yol sürücülerini (tır, otobüs vb.) yorgunluk ve dikkat dağıtıcı davranışlara karşı uyarır.

---

## 📌 Proje Özeti

Bu proje, bilgisayara bağlı bir webcam üzerinden sürücünün yüz hareketlerini analiz ederek:
- Göz kapanmasını
- Esneme davranışını
- Telefon kullanımını
- Sigara kullanımını

gerçek zamanlı olarak tespit eder ve sürücüyü uyarır.

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|-----------|---------------|
| Python | Ana programlama dili |
| OpenCV | Görüntü işleme ve kamera akışı |
| MediaPipe | Yüz landmark tespiti (468 nokta) |
| Ultralytics YOLOv8 | Sigara ve telefon tespiti |
| Streamlit | Web tabanlı dashboard arayüzü |
| Roboflow | Sigara tespiti veri seti |

---

## 🧠 Çalışma Mantığı

### Göz Analizi (EAR - Eye Aspect Ratio)
MediaPipe Face Mesh ile gözlerin 6 landmark noktası belirlenir. Bu noktalar üzerinden EAR değeri hesaplanır:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

- Göz açıksa → EAR yüksek (~0.30)
- Göz kapalıysa → EAR düşük (~0.15)
- EAR < 0.20 belirli süre devam ederse → **DROWSY ALERT**

### Esneme Analizi (MAR - Mouth Aspect Ratio)
Ağız landmark noktaları kullanılarak MAR değeri hesaplanır:
- MAR > 0.60 → **YAWNING DETECTED**

### Nesne Tespiti (YOLOv8)
- **Telefon:** YOLOv8n (COCO dataset, class 67)
- **Sigara:** Özel eğitilmiş YOLOv8n modeli

---

## 📊 Model Eğitimi

Sigara tespiti için Roboflow'dan alınan **8.666 görüntülük** dataset kullanılmıştır.

| Metrik | Değer |
|--------|-------|
| mAP@50 | %92.8 |
| Precision | %91.6 |
| Recall | %86.3 |
| Epoch | 20 |
| Model | YOLOv8n |

**Dataset Split:**
- Train: 6.068 görüntü (%70)
- Validation: 1.726 görüntü (%20)
- Test: 872 görüntü (%10)

---

## 🗂️ Klasör Yapısı

```
driver-monitoring-system/
│
├── app.py                  # Ana Streamlit uygulaması
├── requirements.txt        # Gerekli kütüphaneler
├── README.md
│
├── utils/
│   ├── eye_analysis.py     # EAR hesaplama ve uyuklama tespiti
│   ├── mouth_analysis.py   # MAR hesaplama ve esneme tespiti
│   └── yolo_detection.py   # YOLOv8 sigara/telefon tespiti
│
├── models/
│   └── sigara.pt           # Eğitilmiş sigara tespit modeli
│
└── sounds/                 # Uyarı sesleri (opsiyonel)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlat
```bash
streamlit run app.py
```

### 3. Tarayıcıda Aç
```
http://localhost:8501
```

---

## 🖥️ Arayüz Özellikleri

- **Canlı Kamera:** Webcam görüntüsü üzerinde gerçek zamanlı analiz
- **Durum Paneli:** Anlık sürücü durumu
- **Metrikler:** EAR ve MAR değerleri canlı takip
- **İhlal Sayacı:** Uyuklama, esneme, telefon ve sigara ihlalleri
- **Oturum Süresi:** Toplam sürüş süresi takibi
- **Ayarlar:** EAR/MAR eşik değerleri ve uyarı süresi ayarlanabilir

---

## ⚠️ Tespit Edilen Durumlar

| Durum | Renk | Uyarı |
|-------|------|-------|
| Uyuklama | 🔴 Kırmızı | DROWSY ALERT! |
| Esneme | 🟠 Turuncu | YAWNING DETECTED! |
| Telefon | 🟣 Mor | PHONE DETECTED! |
| Sigara | 🟢 Yeşil | CIGARETTE DETECTED! |

---

## 👨‍💻 Geliştirici

-Muhammet Demir
-ibrahim Uluer
