"""
Streamlit tabanlı karşılaştırmalı nesne tespiti demo arayüzü.

Bu dosya projenin ana giriş noktasıdır. Kullanıcı bu arayüz üzerinden:

1. Görüntü veya video yükler.
2. YOLOv8n ve/veya Faster R-CNN modellerini seçer.
3. Confidence, IoU, hedef sınıf ve video frame örnekleme ayarlarını belirler.
4. Detection sonuçlarını görsel ve tablo halinde inceler.
5. Ground-truth JSON verilirse TP / FP / FN analizi alır.

Çalıştırma:
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Proje içi modüller.
# config.py: sabit sınıf listeleri ve varsayılan eşik değerleri.
# detectors.py: YOLOv8n ve Faster R-CNN model sarmalayıcıları.
# evaluation.py: metrik, model karşılaştırması ve ground-truth değerlendirme fonksiyonları.
# video.py: video frame örnekleme ve annotasyonlu video üretimi.
# visualization.py: bounding box çizimi ve görsel üretim yardımcıları.

from src.config import DEFAULT_CONFIDENCE, DEFAULT_IOU, TRAFFIC_CLASSES
from src.detectors import FasterRCNNDetector, YOLOv8Detector
from src.evaluation import (
    aggregate_stats,
    compare_two_models,
    evaluate_against_ground_truth,
    summarize_detections,
)
from src.video import process_video_file
from src.visualization import draw_detections


st.set_page_config(
    page_title="Object Detection Karşılaştırma Sistemi",
    layout="wide",
)


@st.cache_resource(show_spinner="YOLOv8n modeli yükleniyor...")
def load_yolo(device: str):
    """
    YOLOv8n modelini yükler ve Streamlit cache içinde saklar.
    Object detection modellerinin yüklenmesi zaman alabilir. Streamlit her
    etkileşimde scripti yeniden çalıştırdığı için `st.cache_resource` kullanılır.
    Böylece model bir kez belleğe alınır ve sonraki analizlerde tekrar yüklenmez.
    """
    return YOLOv8Detector(weights="yolov8n.pt", device=device)


@st.cache_resource(show_spinner="Faster R-CNN modeli yükleniyor...")
def load_faster_rcnn(device: str):
    """Faster R-CNN ResNet50-FPN modelini yükler ve cache içinde saklar."""
    return FasterRCNNDetector(device=device)


def load_ground_truth(uploaded_file):
    """
    Yüklenen ground-truth JSON dosyasını okur.

    Beklenen temel format:
        [
            {"class_name": "car", "box": [x1, y1, x2, y2]},
            {"class_name": "traffic light", "box": [x1, y1, x2, y2]}
        ]

    Bazı annotation araçları nesneleri `{"objects": [...]}` formatında
    saklayabildiği için bu format da desteklenir.
    """

    if uploaded_file is None:
        return None

    try:
        data = json.load(uploaded_file)

        # Esnek format desteği: {"objects": [...]} verilirse sadece objects alınır.
        if isinstance(data, dict) and "objects" in data:
            return data["objects"]
        
        # Ana proje formatı doğrudan liste bekler.
        return data

    except Exception as exc:
        # JSON bozuksa uygulama çökmesin; kullanıcıya okunabilir hata verilsin.
        st.error(f"Ground-truth JSON okunamadı: {exc}")
        return None

# Arayüz başlığı ve kısa açıklama.
st.title("Karşılaştırmalı Nesne Tespiti Analiz Sistemi")
st.caption("Senaryo: Trafik ve yol güvenliği | Modeller: YOLOv8n ve Faster R-CNN ResNet50-FPN")

# -----------------------------------------------------------------------------
# Sol menü: kullanıcı ayarları
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")

    # Kullanıcı görüntü veya video analizi arasında seçim yapar.
    input_type = st.radio(
        "Girdi türü",
        ["Görüntü", "Video"],
        horizontal=True,
    )

    # İki modelden biri veya ikisi aynı anda çalıştırılabilir.
    # Karşılaştırmalı analiz için varsayılan olarak ikisi de seçilidir.
    selected_models = st.multiselect(
        "Çalıştırılacak modeller",
        ["YOLOv8n", "Faster R-CNN"],
        default=["YOLOv8n", "Faster R-CNN"],
    )

    # Confidence threshold: düşük seçilirse daha fazla tahmin gelir ama FP artabilir.
    # Yüksek seçilirse model daha seçici olur ancak FN artabilir.
    conf_threshold = st.slider(
        "Confidence threshold",
        0.05,
        0.95,
        DEFAULT_CONFIDENCE,
        0.05,
    )

    # IoU threshold: iki kutunun aynı nesne sayılması için gerekli örtüşme eşiği.
    # Model davranışı karşılaştırması ve ground-truth TP/FP/FN analizinde kullanılır.
    iou_threshold = st.slider(
        "Model eşleştirme IoU threshold",
        0.10,
        0.90,
        DEFAULT_IOU,
        0.05,
    )

    # Cihaz seçimi. "auto" seçilirse CUDA varsa GPU, yoksa CPU kullanılır.
    device = st.selectbox(
        "Cihaz",
        ["auto", "cuda", "cpu"],
        index=0,
    )

    # Trafik senaryosu için ilgilenilen sınıflar.
    # COCO sınıfları arasından trafik/yol güvenliğiyle ilgili olanlar seçilmiştir.
    target_classes = st.multiselect(
        "Trafik senaryosu sınıfları",
        sorted(TRAFFIC_CLASSES),
        default=sorted(TRAFFIC_CLASSES),
    )

    st.divider()

    st.markdown("**Video ayarları**")

    # Video uzun olabilir. Bu nedenle maksimum işlenecek frame sınırlandırılır.
    max_frames = st.number_input(
        "İşlenecek maksimum frame",
        10,
        2000,
        150,
        10,
    )

    # Her frame'i işlemek özellikle Faster R-CNN için çok yavaş olabilir.
    # frame_stride=5 ise 0, 5, 10, 15... numaralı frame'ler işlenir.
    frame_stride = st.number_input(
        "Frame örnekleme aralığı",
        1,
        30,
        5,
        1,
    )

# Model seçilmeden analiz başlatılamaz.
if not selected_models:
    st.warning("En az bir model seçmelisin.")
    st.stop()


# -----------------------------------------------------------------------------
# Seçilen modelleri yükleme
# -----------------------------------------------------------------------------
# Modeller yalnızca seçildiyse yüklenir. Bu, gereksiz bellek ve süre maliyetini azaltır.
detectors = {}

if "YOLOv8n" in selected_models:
    detectors["YOLOv8n"] = load_yolo(device)

if "Faster R-CNN" in selected_models:
    detectors["Faster R-CNN"] = load_faster_rcnn(device)


# -----------------------------------------------------------------------------
# Görüntü analizi modu
# -----------------------------------------------------------------------------
if input_type == "Görüntü":
    uploaded = st.file_uploader(
        "Trafik görüntüsü yükle",
        type=["jpg", "jpeg", "png", "webp"],
    )

    # Ground-truth opsiyoneldir. Verilirse gerçek TP/FP/FN hesabı yapılır.
    # Verilmezse yalnızca iki modelin uyuşmazlıkları FP/FN adayı olarak yorumlanır.
    gt_file = st.file_uploader(
        "Opsiyonel ground-truth JSON yükle",
        type=["json"],
        help='Format: [{"class_name": "car", "box": [x1, y1, x2, y2]}]',
    )

    if uploaded is not None:
        # PIL formatı, hem YOLO hem de Faster R-CNN sarmalayıcıları için ortak giriş olarak kullanılır.
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Orijinal görüntü",
            width="stretch",
        )

        gt = load_ground_truth(gt_file)

        if st.button("Analizi çalıştır", type="primary"):
            # Tüm modellerin çıktıları burada saklanır.
            # İki model seçildiyse sonradan compare_two_models fonksiyonuna verilir.
            all_detections = {}
            stats_rows = []

            # Seçilen model sayısına göre yan yana kolon oluşturulur.
            cols = st.columns(len(detectors))

            for col, (model_name, detector) in zip(cols, detectors.items()):
                # Inference süresi burada ölçülür.
                start = time.perf_counter()
                detections = detector.predict(
                    image,
                    conf_threshold,
                    target_classes,
                )
                elapsed = time.perf_counter() - start

                all_detections[model_name] = detections

                # Detection sayısı, küçük nesne sayısı, ortalama confidence, süre ve FPS hesaplanır.
                stats_rows.append(
                    {
                        "model": model_name,
                        **aggregate_stats(detections, image.size, elapsed),
                    }
                )

                # Görsel üzerine bounding box ve sınıf etiketleri çizilir.
                annotated = draw_detections(
                    image,
                    detections,
                    model_name,
                )

                with col:
                    st.image(
                        annotated,
                        caption=f"{model_name} sonucu",
                        width="stretch",
                    )

                    # Modelin tek tek detection sonuçları tablo halinde gösterilir.
                    st.dataframe(
                        summarize_detections(detections, image.size),
                        width="stretch",
                    )

            st.subheader("FPS ve çalışma süresi karşılaştırması")

            st.dataframe(
                pd.DataFrame(stats_rows),
                width="stretch",
            )

            # İki model varsa, model davranışı karşılaştırması yapılabilir.
            if len(all_detections) == 2:
                model_names = list(all_detections.keys())

                comparison = compare_two_models(
                    all_detections[model_names[0]],
                    all_detections[model_names[1]],
                    iou_threshold,
                )

                st.subheader("Model davranışı karşılaştırması")

                st.write(
                    "matched: iki modelin aynı sınıf ve yeterli IoU ile yakaladığı nesneler. "
                    "model_a_only/model_b_only satırları manuel FP/FN incelemesi için adaydır."
                )

                st.dataframe(
                    comparison,
                    width="stretch",
                )

            # Ground-truth sağlandıysa her model için ayrı TP/FP/FN analizi yapılır.
            if gt is not None:
                st.subheader("Ground-truth tabanlı FP/FN analizi")

                for model_name, detections in all_detections.items():
                    st.markdown(f"**{model_name}**")

                    st.dataframe(
                        evaluate_against_ground_truth(
                            detections,
                            gt,
                            iou_threshold,
                        ),
                        width="stretch",
                    )

# -----------------------------------------------------------------------------
# Video analizi modu
# -----------------------------------------------------------------------------
else:
    uploaded = st.file_uploader(
        "Trafik videosu yükle",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded is not None:
        # Streamlit uploaded_file nesnesi geçici bir dosyaya yazılır.
        # OpenCV VideoCapture dosya yolu istediği için bu adım gereklidir.
        suffix = Path(uploaded.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        # Giriş videosunu tarayıcıya byte olarak vermek, MIME/codec sorunlarını azaltır.
        with open(video_path, "rb") as input_video_file:
            input_video_bytes = input_video_file.read()

        st.subheader("Yüklenen video")
        st.video(input_video_bytes, format="video/mp4")

        if st.button("Video analizini çalıştır", type="primary"):
            try:
                with st.spinner("Video işleniyor. Faster R-CNN seçiliyse işlem yavaş olabilir..."):
                    output_video, stats_df = process_video_file(
                        video_path=video_path,
                        detectors=detectors,
                        conf_threshold=conf_threshold,
                        target_classes=target_classes,
                        max_frames=int(max_frames),
                        frame_stride=int(frame_stride),
                    )

            except Exception as exc:
                # Video codec, OpenCV okuma/yazma veya FFmpeg dönüşüm hataları burada görünür.
                st.error("Video işleme sırasında hata oluştu.")
                st.exception(exc)
                st.stop()

            st.subheader("Annotasyonlu karşılaştırma videosu")

            # Annotasyonlu video da byte olarak gösterilir.
            # Bu yöntem dosya yolu vermeye göre Streamlit içinde daha stabil çalışır.
            with open(output_video, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes, format="video/mp4")

            st.download_button(
                "Annotasyonlu videoyu indir",
                data=video_bytes,
                file_name="annotated_comparison_video.mp4",
                mime="video/mp4",
            )

            st.subheader("Frame bazlı FPS / süre özeti")

            st.dataframe(
                stats_df,
                width="stretch",
            )

            st.subheader("Model bazlı ortalama değerler")

            if not stats_df.empty:
                # Video raporu için en kullanışlı tablo burasıdır.
                # Her model için ortalama detection, küçük nesne, süre ve FPS değerleri hesaplanır.
                summary_df = (
                    stats_df.groupby("model")[
                        ["detections", "small_objects", "runtime_s", "fps_estimate"]
                    ]
                    .mean()
                    .round(3)
                )

                st.dataframe(
                    summary_df,
                    width="stretch",
                )