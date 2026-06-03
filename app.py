import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from utils.eye_analysis import check_drowsiness, LEFT_EYE, RIGHT_EYE, get_eye_landmarks
from utils.mouth_analysis import check_yawning, get_mouth_landmarks, MOUTH_INDICES
from utils.yolo_detection import YOLODetector

# Sayfa ayarları
st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
            /* Streamlit üst barını kaldır */
header[data-testid="stHeader"] {
    display: none;
}

/* Sağ üst toolbar */
[data-testid="stToolbar"] {
    display: none;
}
/* Sidebar kapatma butonunu kaldır */
[data-testid="collapsedControl"] {
    display: none !important;
}
/* Sidebar'ın küçültülmesini engelle */
[data-testid="stSidebar"] button {
    display: none !important;
}

/* Deploy butonu */
.stDeployButton {
    display: none;
}

/* Hamburger menü */
#MainMenu {
    visibility: hidden;
}

/* Footer */
footer {
    visibility: hidden;
}

/* Header boşluğu */
header {
    visibility: hidden;
    height: 0px;
}

/* Üst boşluğu azalt */
.block-container {
    padding-top: 1rem;
}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*{
    font-family:'Inter',sans-serif;
}

/* Ana Arkaplan */
.stApp{
    background:
    radial-gradient(circle at top left,#1e3a8a 0%,transparent 35%),
    radial-gradient(circle at bottom right,#0891b2 0%,transparent 35%),
    linear-gradient(135deg,#050816,#0f172a,#111827);
}

/* Genel container */
.block-container{
    padding-top:1rem;
}

/* Başlık */
h1{
    color:white !important;
    text-align:center;
    font-size:2.3rem !important;
    font-weight:700 !important;
    letter-spacing:-1px;
}

h2,h3{
    color:#dbeafe !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:rgba(15,23,42,0.75);
    backdrop-filter:blur(20px);
    border-right:1px solid rgba(255,255,255,0.08);
}

/* Cam Kartlar */
[data-testid="stMetric"]{
    background:rgba(255,255,255,0.05);
    backdrop-filter:blur(20px);

    border:1px solid rgba(255,255,255,0.08);

    border-radius:18px;

    padding:18px;

    transition:all .3s ease;
}

[data-testid="stMetric"]:hover{
    transform:translateY(-3px);
    border-color:#38bdf8;
    box-shadow:
    0 0 25px rgba(56,189,248,.25);
}

/* Metric Label */
[data-testid="stMetricLabel"]{
    color:#94a3b8 !important;
    text-transform:uppercase;
    letter-spacing:1px;
    font-size:.75rem !important;
}

/* Metric Value */
[data-testid="stMetricValue"]{
    color:white !important;
    font-size:1.7rem !important;
    font-weight:700 !important;
}

/* Butonlar */
.stButton button{

    width:100%;

    background:linear-gradient(
    135deg,
    #2563eb,
    #06b6d4);

    color:white !important;

    border:none !important;

    border-radius:14px !important;

    padding:.8rem !important;

    font-weight:600 !important;

    transition:.3s;
}

.stButton button:hover{

    transform:translateY(-2px);

    box-shadow:
    0 0 25px rgba(59,130,246,.4);

}

/* Kamera alanı */
[data-testid="stImage"]{

    border-radius:24px;

    overflow:hidden;

    border:2px solid rgba(56,189,248,.25);

    box-shadow:
    0 0 35px rgba(56,189,248,.18);

}

/* Alert Kutuları */
[data-testid="stAlert"]{

    border-radius:16px !important;

    backdrop-filter:blur(20px);

    border:none !important;

}

/* Inputlar */
input{

    background:rgba(255,255,255,.05) !important;

    color:white !important;

    border-radius:12px !important;

    border:1px solid rgba(255,255,255,.08) !important;

}

/* Slider */
[data-baseweb="slider"]{
    padding-top:15px;
}

/* Scroll */
::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#38bdf8;
    border-radius:20px;
}

::-webkit-scrollbar-track{
    background:transparent;
}

/* Yavaş parlayan efekt */
@keyframes glow{
    0%{
        box-shadow:0 0 10px rgba(56,189,248,.2);
    }
    50%{
        box-shadow:0 0 30px rgba(56,189,248,.4);
    }
    100%{
        box-shadow:0 0 10px rgba(56,189,248,.2);
    }
}

[data-testid="stMetric"]{
    animation:glow 4s infinite;
}

/* Divider */
hr{
    border-color:rgba(255,255,255,.08);
}
</style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("""
<div style='text-align:center; padding-bottom:20px'>
    <h1 style='margin-bottom:0'>
        🚘 Driver Monitoring System
    </h1>
    <p style='color:#94a3b8;font-size:20px'>
        AI-Powered Drowsiness & Distraction Detection
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar ayarları
st.sidebar.title("⚙️ Ayarlar")
EAR_THRESHOLD = st.sidebar.slider("Göz Kapanma Eşiği (EAR)", 0.10, 0.35, 0.20, 0.01)
MAR_THRESHOLD = st.sidebar.slider("Esneme Eşiği (MAR)\n [0.40 Önerilir.]", 0.3, 0.9, 0.6, 0.05)
DROWSY_FRAMES = st.sidebar.slider(
    "Uyarı için gereken frame sayısı",
    10,
    40,
    20,
    5
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚬 Sigara Tespiti")

use_custom_model = st.sidebar.checkbox(
    "Sigara Tespitini Aktifleştir",
    value=True
)
if use_custom_model:
    st.sidebar.success("Sigara algılama sistemi aktif.")
else:
    st.sidebar.warning("Sigara algılama sistemi pasif.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Eşik Değerleri Hakkında")
st.sidebar.info(
    f"EAR < {EAR_THRESHOLD:.2f} → Göz kapalı\n"
    f"MAR > {MAR_THRESHOLD:.2f} → Esneme\n"
    f"{DROWSY_FRAMES} frame üst üste göz kapalıysa uyarı"
)

# Ana layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Canlı Kamera")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Durum Paneli")
    status_placeholder = st.empty()
    
    st.markdown("### 📈 Metrikler")
    ear_placeholder = st.empty()
    mar_placeholder = st.empty()
    
    st.markdown("### ⚠️ İhlal Sayacı")
    drowsy_count_ph = st.empty()
    yawn_count_ph = st.empty()
    phone_count_ph = st.empty()
    cigarette_count_ph = st.empty()
    
    st.markdown("### 🕐 Oturum")
    session_ph = st.empty()

# Kontrol butonları
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    start_btn = st.button("▶️ Kamerayı Başlat", type="primary", use_container_width=True)
with col_btn2:
    stop_btn = st.button("⏹️ Durdur", use_container_width=True)

# Session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'drowsy_count' not in st.session_state:
    st.session_state.drowsy_count = 0
if 'yawn_count' not in st.session_state:
    st.session_state.yawn_count = 0
if 'phone_count' not in st.session_state:
    st.session_state.phone_count = 0
if 'cigarette_count' not in st.session_state:
    st.session_state.cigarette_count = 0
if 'phone_frame_count' not in st.session_state:
    st.session_state.phone_frame_count = 0
if 'cig_frame_count' not in st.session_state:
    st.session_state.cig_frame_count = 0
if 'alarm_playing' not in st.session_state:
    st.session_state.alarm_playing = False
if 'phone_alarm_playing' not in st.session_state:
    st.session_state.phone_alarm_playing = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# MediaPipe kurulumu
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# YOLO kurulumu
@st.cache_resource
def load_yolo(custom_path=None, use_custom=False):
    detector = YOLODetector('yolov8n.pt')
    if use_custom and custom_path:
        detector.load_custom_model(custom_path)
    return detector

DEFAULT_CIG_MODEL = "models/sigara.pt"

detector = load_yolo(
    DEFAULT_CIG_MODEL,
    use_custom_model
)

# Pygame ses sistemi
try:
    pygame.mixer.init()

    alarm_sound = pygame.mixer.Sound("assets/alarm.mp3")
    phone_alarm_sound = pygame.mixer.Sound("assets/alarm_phone.mp3")

    sound_available = True

except Exception as e:
    print(e)
    sound_available = False

# Ana döngü
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Kamera açılamadı! Kameranızı kontrol edin.")
        st.session_state.running = False
    else:
        drowsy_counter = 0
        yawn_counter = 0
        prev_drowsy = False
        prev_yawn = False
        start_time = time.time()
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Kameradan görüntü alınamadı.")
                    break
                
                frame = cv2.flip(frame, 1)
                img_h, img_w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = face_mesh.process(rgb_frame)
                
                drowsy_alert = False
                yawn_alert = False
                left_ear = 0.0
                right_ear = 0.0
                mar = 0.0
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    drowsy_alert, drowsy_counter, left_ear, right_ear = check_drowsiness(
                        face_landmarks,
                        img_w,
                        img_h,
                        EAR_THRESHOLD,
                        DROWSY_FRAMES,
                        drowsy_counter
                    )
                    
                    # Esneme analizi
                    yawn_alert, yawn_counter, mar = check_yawning(
                        face_landmarks, img_w, img_h,
                        MAR_THRESHOLD, yawn_counter
                    )
                    
                    # Göz noktalarını çiz
                    for idx in LEFT_EYE + RIGHT_EYE:
                        lm = face_landmarks.landmark[idx]
                        x = int(lm.x * img_w)
                        y = int(lm.y * img_h)
                        cv2.circle(rgb_frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Ağız noktalarını çiz
                    for idx in MOUTH_INDICES:
                        lm = face_landmarks.landmark[idx]
                        x = int(lm.x * img_w)
                        y = int(lm.y * img_h)
                        cv2.circle(rgb_frame, (x, y), 2, (255, 255, 0), -1)
                    
                    # Uyuklama/esneme sayaçlarını güncelle
                    if drowsy_alert and not prev_drowsy:
                        st.session_state.drowsy_count += 1
                    if yawn_alert and not prev_yawn:
                        st.session_state.yawn_count += 1
                    
                    prev_drowsy = drowsy_alert
                    prev_yawn = yawn_alert
                
                else:
                    cv2.putText(rgb_frame, "YUZ TESPIT EDILEMIYOR",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 165, 255), 2)
                
                # YOLO tespiti
                yolo_frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                yolo_frame_bgr, detections = detector.detect(yolo_frame_bgr)
                rgb_frame = cv2.cvtColor(yolo_frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Telefon ve sigara frame sayacı (kopmaları önler)
                phone_detected = any(d['label'] == 'cell phone' for d in detections)
                cigarette_detected = any(d['label'] == 'cigarette' for d in detections)

                if phone_detected:
                    st.session_state.phone_frame_count += 1
                else:
                    st.session_state.phone_frame_count = 0

                if cigarette_detected:
                    st.session_state.cig_frame_count += 1
                else:
                    st.session_state.cig_frame_count = 0
                # 10 frame üst üste tespit edilirse sayacı 1 artır
                if st.session_state.phone_frame_count == 10:
                    st.session_state.phone_count += 1
                if st.session_state.cig_frame_count == 10:
                    st.session_state.cigarette_count += 1

                # Alarm sistemi
                
                # Uyarı overlay - göz kapanma
                if drowsy_alert:
                    print("UYUKLAMA ALGILANDI")
                    if sound_available and not st.session_state.alarm_playing:
                        alarm_sound.play(-1)
                        st.session_state.alarm_playing = True

                    cv2.rectangle(rgb_frame, (0, 0), (img_w, img_h), (255, 0, 0), 8)
                    cv2.putText(rgb_frame, "DROWSY ALERT!",
                        (int(img_w/2) - 150, int(img_h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

                else:
                    if sound_available and st.session_state.alarm_playing:
                        alarm_sound.stop()
                    st.session_state.alarm_playing = False
                
                # Uyarı overlay - esneme
                if yawn_alert:
                    cv2.rectangle(rgb_frame, (0, 0), (img_w, img_h), (255, 165, 0), 8)
                    cv2.putText(rgb_frame, "YAWNING DETECTED!",
                               (int(img_w/2) - 180, int(img_h/2) + 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)

                # Uyarı overlay - telefon
                if st.session_state.phone_frame_count >= 10:

                    if sound_available and not st.session_state.phone_alarm_playing:
                        phone_alarm_sound.play(-1)
                        st.session_state.phone_alarm_playing = True

                else:

                    if sound_available and st.session_state.phone_alarm_playing:
                        phone_alarm_sound.stop()
                        st.session_state.phone_alarm_playing = False
                if st.session_state.phone_frame_count >= 10:
                    cv2.rectangle(rgb_frame, (0, 0), (img_w, img_h), (128, 0, 128), 8)
                    cv2.putText(rgb_frame, "PHONE DETECTED!",
                               (int(img_w/2) - 160, int(img_h/2) + 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 0, 128), 3)

                # Uyarı overlay - sigara
                if st.session_state.cig_frame_count >= 10:
                    cv2.rectangle(rgb_frame, (0, 0), (img_w, img_h), (0, 180, 0), 8)
                    cv2.putText(rgb_frame, "CIGARETTE DETECTED!",
                               (int(img_w/2) - 190, int(img_h/2) + 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)
                
                # EAR/MAR değerlerini frame üzerine yaz
                cv2.putText(rgb_frame, f"EAR: {(left_ear+right_ear)/2:.2f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(rgb_frame, f"MAR: {mar:.2f}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Frame'i göster
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Durum panelini güncelle
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                
                if drowsy_alert:
                    status_placeholder.error("😴 UYUKLAMA TESPİT EDİLDİ!")
                elif yawn_alert:
                    status_placeholder.warning("🥱 ESNEME TESPİT EDİLDİ!")
                elif st.session_state.phone_frame_count >= 10:
                    status_placeholder.error("📱 TELEFON KULLANIMI TESPİT EDİLDİ!")
                elif st.session_state.cig_frame_count >= 10:
                    status_placeholder.error("🚬 SİGARA KULLANIMI TESPİT EDİLDİ!")
                elif results.multi_face_landmarks:
                    status_placeholder.success("✅ Sürücü Aktif")
                else:
                    status_placeholder.warning("⚠️ Yüz Tespit Edilemiyor")
                
                ear_placeholder.metric("Ortalama EAR", f"{(left_ear+right_ear)/2:.3f}")
                mar_placeholder.metric("MAR", f"{mar:.3f}")
                
                drowsy_count_ph.metric("😴 Uyuklama", st.session_state.drowsy_count)
                yawn_count_ph.metric("🥱 Esneme", st.session_state.yawn_count)
                phone_count_ph.metric("📱 Telefon", st.session_state.phone_count)
                cigarette_count_ph.metric("🚬 Sigara", st.session_state.cigarette_count)
                
                session_ph.metric("⏱️ Süre", f"{mins:02d}:{secs:02d}")
        
        cap.release()