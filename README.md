# Karşılaştırmalı Nesne Tespiti Analiz Sistemi

Bu proje, **Bilgisayarla Görü** dersi final ödevi kapsamında geliştirilmiş karşılaştırmalı bir nesne tespiti analiz sistemidir. Çalışmada trafik ve yol güvenliği senaryosu seçilmiş; aynı görüntü ve video girdileri üzerinde **YOLOv8n** ve **Faster R-CNN ResNet50-FPN** modellerinin davranışları karşılaştırılmıştır.

Proje yalnızca hazır model çıktısı göstermeyi değil; modellerin küçük nesne, kalabalık sahne, yanlış pozitif/yanlış negatif, FPS ve çalışma süresi açısından teknik olarak analiz edilmesini amaçlamaktadır.

---

## Proje Amacı

Modern object detection mimarileri gerçek dünya trafik sahnelerinde farklı davranışlar gösterebilir. Uzak trafik ışıkları, yoğun araç trafiği, gece far parlamaları, kısmen örtülü yayalar ve düşük çözünürlüklü video sahneleri modellerin hata üretmesine neden olabilir.

Bu proje kapsamında geliştirilen sistem:

- Aynı görüntü veya video üzerinde iki farklı object detection modelini çalıştırır.
- Detection sonuçlarını bounding box, sınıf adı ve confidence değeriyle görselleştirir.
- Modellerin çıktılarını sayısal ve görsel olarak karşılaştırır.
- FPS ve çalışma süresi değerlerini hesaplar.
- Opsiyonel ground-truth JSON ile TP / FP / FN analizi yapar.
- Streamlit tabanlı çalıştırılabilir bir demo arayüzü sunar.

---

## Kullanılan Modeller

### YOLOv8n

YOLOv8n, YOLO ailesinin hafif ve hızlı bir varyantıdır. Tek aşamalı nesne tespiti yaklaşımına dayanır. Görüntü üzerinden doğrudan bounding box, sınıf ve confidence tahmini üretir. Bu nedenle gerçek zamanlı veya gerçek zamana yakın uygulamalar için uygundur.

Bu projede YOLOv8n modeli özellikle:

- FPS performansı,
- hızlı inference davranışı,
- küçük nesne kaçırma eğilimi,
- yoğun sahnelerde NMS etkisi

açısından değerlendirilmiştir.

### Faster R-CNN ResNet50-FPN

Faster R-CNN iki aşamalı bir object detection mimarisidir. İlk aşamada Region Proposal Network ile aday nesne bölgeleri üretilir, ikinci aşamada bu bölgeler sınıflandırılır ve bounding box regresyonu uygulanır.

Bu projede Faster R-CNN ResNet50-FPN modeli özellikle:

- region proposal tabanlı davranış,
- küçük nesne adayları,
- lokalizasyon davranışı,
- yüksek işlem maliyeti

açısından YOLOv8n ile karşılaştırılmıştır.

---

## Özellikler

- Görüntü yükleme desteği: `JPG`, `JPEG`, `PNG`, `WEBP`
- Video yükleme desteği: `MP4`, `AVI`, `MOV`, `MKV`
- YOLOv8n ve Faster R-CNN model seçimi
- Confidence threshold ayarı
- IoU threshold ayarı
- Hedef sınıf seçimi
- Ground-truth JSON yükleme desteği
- Görsel bounding box çizimi
- Yan yana model karşılaştırması
- Model davranışı karşılaştırması
- TP / FP / FN analizi
- FPS ve çalışma süresi hesaplama
- Annotasyonlu karşılaştırma videosu üretme
- CSV çıktıları ile raporlamaya uygun analiz sonuçları

---

## Kullanılan Veri Seti

Projede trafik ve yol güvenliği senaryosunu temsil eden gerçek görüntü ve video girdileri kullanılmıştır.

| Dosya | Tür | Analiz Amacı |
|---|---|---|
| `small_traffic_light.jpg` | Görüntü | Küçük trafik ışığı ve uzak nesne davranışı |
| `city_traffic_crowded.jpg` | Görüntü | Kalabalık araç sahnesi ve yoğun trafik analizi |
| `night_glare_traffic.jpg` | Görüntü | Gece, düşük ışık ve far/parlama etkisi |
| `occluded_pedestrian.jpg` | Görüntü | Kısmen örtülü yaya ve kalabalık insan sahnesi |
| `dense_vehicle_scene.mp4` | Video | Yoğun araç videosu ve FPS analizi |
| `vehicle_scene_2.mp4` | Video | Düşük çözünürlük/FPS koşullarında araç tespiti |
| `mix_scene.mp4` | Video | Karma sahnede video tabanlı detection kararlılığı |

---

## Sistem Mimarisi

Sistem aşağıdaki temel bileşenlerden oluşmaktadır:

| Bileşen | Görev |
|---|---|
| `app.py` | Streamlit arayüzü, dosya yükleme, model seçimi, sonuç gösterimi |
| `src/detectors.py` | YOLOv8n ve Faster R-CNN model sarmalayıcıları |
| `src/evaluation.py` | IoU, küçük nesne kontrolü, model eşleşmesi, TP / FP / FN analizi |
| `src/visualization.py` | Bounding box çizimi ve yan yana görselleştirme |
| `src/video.py` | Video frame örnekleme, model çalıştırma, annotasyonlu video üretimi |
| `scripts/run_image_batch.py` | Klasör bazlı toplu görüntü analizi ve CSV üretimi |

Genel akış:

```text
Girdi
  ├── Görüntü
  ├── Video
  └── Ground-truth JSON
        ↓
Streamlit Arayüzü
        ↓
Model Sarmalayıcıları
  ├── YOLOv8n
  └── Faster R-CNN ResNet50-FPN
        ↓
Ortak Detection Formatı
        ↓
Değerlendirme + Görselleştirme + Video İşleme
        ↓
Analiz Çıktıları
  ├── Detection tabloları
  ├── FPS / süre karşılaştırmaları
  ├── Model davranışı karşılaştırması
  ├── TP / FP / FN sonuçları
  └── Annotasyonlu görsel/video
```

---

## Kurulum

Bu proje için önerilen Python sürümü:

```text
Python 3.10
```

Windows PowerShell üzerinde sanal ortam oluşturmak için:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Eğer PowerShell çalıştırma izni hatası verirse:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Paketleri kurmak için:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Çalıştırma

Streamlit arayüzünü başlatmak için:

```powershell
python -m streamlit run app.py
```

Uygulama varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

---

## Kullanım

### Görüntü Analizi

1. Arayüzden `Görüntü` modunu seç.
2. Bir trafik görüntüsü yükle.
3. İstenirse ground-truth JSON dosyası yükle.
4. YOLOv8n ve Faster R-CNN modellerini seç.
5. Confidence ve IoU threshold değerlerini ayarla.
6. `Analizi çalıştır` butonuna bas.
7. Aşağıdaki çıktıları incele:
   - Annotasyonlu detection görselleri
   - Detection tabloları
   - FPS ve çalışma süresi karşılaştırması
   - Model davranışı karşılaştırması
   - TP / FP / FN analizi

### Video Analizi

1. Arayüzden `Video` modunu seç.
2. Video dosyasını yükle.
3. İşlenecek maksimum frame sayısını ve frame stride değerini belirle.
4. `Video analizini çalıştır` butonuna bas.
5. Aşağıdaki çıktıları incele:
   - Annotasyonlu karşılaştırma videosu
   - Frame bazlı FPS / süre tablosu
   - Model bazlı ortalama değerler
   - Annotasyonlu video indirme çıktısı

---

## Ground-truth JSON Formatı

Ground-truth JSON dosyası, görüntüdeki gerçek nesne konumlarını belirtmek için kullanılır. Format aşağıdaki gibidir:

```json
[
  {
    "class_name": "car",
    "box": [120, 80, 300, 220]
  },
  {
    "class_name": "traffic light",
    "box": [520, 40, 555, 115]
  }
]
```

Bounding box formatı:

```text
[x1, y1, x2, y2]
```

Burada:

- `x1`: sol üst köşe x koordinatı
- `y1`: sol üst köşe y koordinatı
- `x2`: sağ alt köşe x koordinatı
- `y2`: sağ alt köşe y koordinatı

Desteklenen örnek sınıflar:

```text
car
bus
truck
person
traffic light
motorcycle
bicycle
```

Ground-truth JSON yüklendiğinde sistem model tahminlerini gerçek etiketlerle IoU tabanlı olarak karşılaştırır ve şu değerleri hesaplar:

- **TP**: True Positive
- **FP**: False Positive
- **FN**: False Negative

---

## Analiz Metrikleri

Projede kullanılan başlıca analiz ölçütleri şunlardır:

| Metrik | Açıklama |
|---|---|
| Detection Sayısı | Modelin ürettiği toplam tespit sayısı |
| Küçük Nesne Sayısı | Görüntü alanına göre küçük kabul edilen bounding box sayısı |
| Ortalama Confidence | Modelin tespitleri için ortalama güven skoru |
| Süre (s) | Modelin inference süresi |
| FPS | Saniye başına işlenebilecek tahmini frame sayısı |
| Matched | İki modelin aynı sınıf ve yeterli IoU ile eşleşen tahminleri |
| FP | Modelin gerçekte olmayan nesne tahmini |
| FN | Görüntüde bulunan fakat modelin kaçırdığı nesne |

---

## Deneysel Analiz Başlıkları

Bu projede aşağıdaki analizler gerçekleştirilmiştir:

### 1. Küçük Nesne Davranışı

`small_traffic_light.jpg` görüntüsü küçük trafik ışıkları ve uzak nesneler için kullanılmıştır. Amaç, modellerin düşük piksel alanına sahip nesneleri tespit etme başarısını incelemektir.

### 2. Kalabalık Sahne Performansı

`city_traffic_crowded.jpg` görüntüsü yoğun araç trafiği içermektedir. Bu sahne, üst üste binen kutular, NMS etkisi, localization error ve duplicate detection davranışlarını incelemek için kullanılmıştır.

### 3. Gece ve Parlama Koşulları

`night_glare_traffic.jpg` görüntüsü düşük ışık ve far/parlama etkilerini analiz etmek için kullanılmıştır. Bu sahnede parlak bölgelerin false positive üretme ihtimali değerlendirilmiştir.

### 4. Kısmen Örtülü Yaya Analizi

`occluded_pedestrian.jpg` görüntüsü kalabalık yaya sahnesi ve occlusion davranışı için kullanılmıştır. Kısmen görünen insanların modeller tarafından kaçırılma eğilimi incelenmiştir.

### 5. Video Tabanlı FPS ve Kararlılık Analizi

Üç farklı video girdisi üzerinde modellerin frame bazlı çalışma süresi, ortalama FPS değeri ve detection kararlılığı analiz edilmiştir.

---

## Başarısız Durum Analizleri

Projede en az beş başarısız durum teknik olarak incelenmiştir:

| No | Sahne | Hata Türü | Teknik Neden |
|---:|---|---|---|
| 1 | Uzak trafik ışığı | False Negative | Nesne alanının küçük olması ve düşük piksel temsili |
| 2 | Yoğun araç trafiği | Localization error / duplicate | Araçların üst üste binmesi ve NMS/proposal etkisi |
| 3 | Gece far/parlama | False Positive | Parlak bölgelerin nesne gibi algılanması |
| 4 | Kısmen örtülü yaya | False Negative | İnsan siluetinin eksik görünmesi |
| 5 | Video sahnesi | Frame bazlı tutarsız detection | Modellerin temporal bilgi kullanmaması |

---

## Proje Klasör Yapısı

Önerilen klasör yapısı:

```text
traffic_object_detection_comparison_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── detectors.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── video.py
│
├── scripts/
│   └── run_image_batch.py
│
├── data/
│   ├── samples/
│   └── annotations/
│
├── outputs/
│   ├── images/
│   ├── videos/
│   └── csv/
│
└── docs/
    └── technical_report.docx
```

---

## Gereksinimler

Temel kütüphaneler:

```text
streamlit
ultralytics
torch
torchvision
opencv-python
pillow
numpy
pandas
matplotlib
imageio-ffmpeg
```

Paketler `requirements.txt` dosyası üzerinden kurulmalıdır.

---

## Notlar

- İlk çalıştırmada modellerin ağırlıkları indirilebilir; bu nedenle ilk inference daha uzun sürebilir.
- CUDA destekli GPU varsa işlem süresi azalabilir.
- Faster R-CNN modeli YOLOv8n’e göre daha yavaş çalışır.
- Video analizinde frame sayısı ve frame stride değeri işlem süresini doğrudan etkiler.
- Ground-truth JSON kullanılmadan kesin TP / FP / FN hesabı yapılamaz; ground-truth olmadığında yalnızca model uyuşmazlığı adayları incelenebilir.

---

## Teslim İçeriği

Bu proje aşağıdaki teslim bileşenlerini içerir:

- GitHub repository
- Streamlit tabanlı çalıştırılabilir demo
- Teknik rapor
- Görüntü ve video analiz çıktıları
- Ground-truth JSON örnekleri
- FPS, süre, FP/FN ve failure case analizleri

---

## Çalıştırma Özeti

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Geliştirici

Bu proje, Bilgisayar Mühendisliği Ana Bilim Dalı Bilgisayarla Görü dersi final projesi kapsamında hazırlanmıştır.

**Senaryo:** Trafik ve Yol Güvenliği  
**Modeller:** YOLOv8n, Faster R-CNN ResNet50-FPN  
**Arayüz:** Streamlit  
**Analiz Türleri:** Görüntü analizi, video analizi, FP/FN analizi, FPS karşılaştırması, failure case incelemesi
