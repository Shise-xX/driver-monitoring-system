import numpy as np

# Göz landmark indexleri (MediaPipe Face Mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_EAR(eye_landmarks):
    """
    Eye Aspect Ratio hesaplar.
    Göz açıksa yüksek, kapalıysa düşük değer döner.
    """
    # Dikey mesafeler
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Yatay mesafe
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def get_eye_landmarks(face_landmarks, eye_indices, img_w, img_h):
    """
    Belirtilen landmark indexlerinden koordinat dizisi döner.
    """
    coords = []
    for idx in eye_indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    return np.array(coords)

def check_drowsiness(face_landmarks, img_w, img_h, EAR_THRESHOLD=0.20, frame_count=20, counter=0):
    """
    Uyuklama tespiti yapar.
    Döner: (uyarı_var_mı, güncel_counter, sol_EAR, sağ_EAR)
    """
    left = get_eye_landmarks(face_landmarks, LEFT_EYE, img_w, img_h)
    right = get_eye_landmarks(face_landmarks, RIGHT_EYE, img_w, img_h)
    
    left_ear = calculate_EAR(left)
    right_ear = calculate_EAR(right)
    avg_ear = (left_ear + right_ear) / 2.0
    
    if avg_ear < EAR_THRESHOLD:
        counter += 1
    else:
        counter = 0
    
    # 20 frame üst üste kapalıysa uyarı ver (~0.6 sn @ 30fps)
    alert = counter >= frame_count
    
    return alert, counter, left_ear, right_ear