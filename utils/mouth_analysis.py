import numpy as np

# Ağız landmark indexleri (MediaPipe Face Mesh)
MOUTH_INDICES = [13, 14, 78, 308, 82, 312]

def calculate_MAR(mouth_landmarks):
    """
    Mouth Aspect Ratio hesaplar.
    Ağız açıksa yüksek, kapalıysa düşük değer döner.
    """
    # Dikey mesafeler
    A = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
    B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[5])
    # Yatay mesafe
    C = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
    
    mar = (A + B) / (2.0 * C)
    return mar

def get_mouth_landmarks(face_landmarks, img_w, img_h):
    """
    Ağız landmark koordinatlarını döner.
    """
    coords = []
    for idx in MOUTH_INDICES:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    return np.array(coords)

def check_yawning(face_landmarks, img_w, img_h, MAR_THRESHOLD=0.6, counter=0):
    """
    Esneme tespiti yapar.
    Döner: (uyarı_var_mı, güncel_counter, mar_değeri)
    """
    mouth = get_mouth_landmarks(face_landmarks, img_w, img_h)
    mar = calculate_MAR(mouth)
    
    if mar > MAR_THRESHOLD:
        counter += 1
    else:
        counter = 0
    
    # 15 frame üst üste açıksa esneme sayılır (~0.5 sn @ 30fps)
    alert = counter >= 15
    
    return alert, counter, mar