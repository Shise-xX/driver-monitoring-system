from ultralytics import YOLO
import cv2

# Tespit etmek istediğimiz sınıflar (COCO dataset sınıf isimleri)
TARGET_CLASSES = {
    'cell phone': 'TELEFON KULLANIYOR!',
    'cigarette': 'SIGARA ICIYOR!'
}

# COCO'da cigarette yok, bunun için ayrı model gerekecek
# Şimdilik telefon tespiti için COCO kullanalım
COCO_TARGETS = {
    67: 'cell phone'  # COCO class id
}

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        YOLOv8 modelini yükler.
        model_path: eğitilmiş model yolu veya varsayılan yolov8n.pt
        """
        self.model = YOLO(model_path)
        self.custom_model = None
    
    def load_custom_model(self, model_path):
        """
        Sigara tespiti için eğitilmiş özel modeli yükler.
        """
        try:
            self.custom_model = YOLO(model_path)
            print(f"Özel model yüklendi: {model_path}")
            return True
        except Exception as e:
            print(f"Model yüklenemedi: {e}")
            return False
    
    def detect(self, frame):
        """
        Verilen frame üzerinde tespit yapar.
        Döner: (işlenmiş_frame, tespit_listesi)
        """
        detections = []
        
        # Telefon tespiti (COCO modeli)
        results = self.model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if class_id in COCO_TARGETS and conf > 0.5:
                    label = COCO_TARGETS[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Kutu çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}',
                               (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 0, 255), 2)
                    
                    detections.append({
                        'label': label,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        # Sigara tespiti (özel model varsa)
        if self.custom_model:
            custom_results = self.custom_model(frame, verbose=False)
            for result in custom_results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(frame, f'Sigara {conf:.2f}',
                                   (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (0, 165, 255), 2)
                        detections.append({
                            'label': 'cigarette',
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })
        
        return frame, detections