"""
Proje genelinde kullanılan sabit ayarlar.

Bu dosya, farklı modüllerde tekrar tekrar kullanılacak değerleri merkezi bir
noktada toplar. Böylece sınıf listesi veya threshold değerleri değiştirileceği
zaman tek dosyadan yönetilebilir.
"""

# COCO veri setindeki sınıflardan trafik ve yol güvenliği senaryosu için anlamlı
# olanlar seçilmiştir. Model çıktıları bu sınıflarla filtrelenir.
TRAFFIC_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}

# Varsayılan confidence threshold.
# Düşük değer daha fazla detection üretir fakat false positive sayısını artırabilir.
# Yüksek değer modeli daha seçici yapar fakat false negative riskini artırabilir.
DEFAULT_CONFIDENCE = 0.30

# Varsayılan IoU threshold.
# İki bounding box'ın aynı nesne olarak kabul edilmesi için gereken örtüşme eşiğidir.
DEFAULT_IOU = 0.50

# Küçük nesne kontrolünde kullanılan alan oranı.
# Bir bounding box görüntü alanının %1'inden küçükse küçük nesne kabul edilir.
SMALL_OBJECT_AREA_RATIO = 0.01  # bbox area <= 1% image area

# COCO literatüründe küçük nesne için referans alan 32x32 piksel olarak alınır.
# Bu projede küçük nesne işaretlemede pratik eşiklerden biri olarak kullanılır.
COCO_SMALL_AREA_PX = 32 * 32     # COCO-style small object reference

# Rapor veya arayüzde model adlarını daha açıklayıcı göstermek için kullanılan sözlük.
MODEL_DISPLAY_NAMES = {
    "yolov8n": "YOLOv8n - single-stage detector",
    "fasterrcnn": "Faster R-CNN ResNet50-FPN - two-stage detector",
}
