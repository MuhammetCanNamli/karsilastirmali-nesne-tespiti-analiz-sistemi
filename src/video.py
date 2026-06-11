"""
Video işleme yardımcı fonksiyonları.

Bu modül, video girdilerinde iki modelin frame bazlı çalıştırılması ve
annotasyonlu karşılaştırma videosu üretilmesi için kullanılır.

Ana görevler:
    - OpenCV ile video okuma
    - Belirli aralıklarla frame örnekleme
    - Her frame üzerinde seçili detector modellerini çalıştırma
    - Detection sonuçlarını yan yana görselleştirme
    - Frame bazlı FPS/süre tablosu üretme
    - Tarayıcı uyumlu H.264 MP4 çıktı oluşturma
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Dict, Iterable, List, Tuple

import cv2
import pandas as pd

from .evaluation import aggregate_stats
from .visualization import bgr_to_pil, concat_side_by_side, draw_detections, pil_to_bgr


def _convert_to_browser_compatible_mp4(input_path: str, output_path: str) -> str:
    """
    OpenCV'nin ürettiği videoyu tarayıcı uyumlu H.264 MP4 formatına dönüştürür.

    OpenCV çoğu ortamda MP4 dosyasını `mp4v` codec ile yazar. Bu dosya teknik
    olarak geçerli olsa bile Streamlit/Chrome/Firefox içinde oynatılmayabilir.
    Bu nedenle video FFmpeg ile H.264 + yuv420p formatına yeniden encode edilir.

    Args:
        input_path: OpenCV tarafından üretilen ham MP4 dosya yolu.
        output_path: H.264 olarak üretilecek final MP4 dosya yolu.

    Returns:
        Tarayıcı uyumlu video dosya yolu.

    Raises:
        RuntimeError: FFmpeg dönüşümü başarısız olursa hata mesajı ile yükseltilir.
    """
    try:
        # imageio-ffmpeg, sisteme ayrıca ffmpeg kurmadan paket içinden ffmpeg sağlar.
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # imageio_ffmpeg yoksa sistem PATH içindeki ffmpeg denenir.
        ffmpeg_exe = "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-y", # Var olan çıktı dosyasının üzerine yaz.
        "-i",
        input_path,
        "-an", # Ses kanalı gerekmediği için kaldırılır.
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2", # H.264 için genişlik/yükseklik çift sayı olsun.
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p", # Tarayıcı uyumluluğu için en güvenli piksel formatlarından biri.
        "-movflags",
        "+faststart", # Web oynatma için metadata dosya başına taşınır.
        "-preset",
        "veryfast",
        "-crf",
        "23",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg video conversion failed:\n"
            + result.stderr[-2000:]
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Converted video file was not created correctly.")

    return output_path


def process_video_file(
    video_path: str,
    detectors: Dict[str, object],
    conf_threshold: float,
    target_classes: Iterable[str],
    max_frames: int = 300,
    frame_stride: int = 5,
) -> Tuple[str, pd.DataFrame]:
    """
    Video üzerinde seçili detector modellerini çalıştırır.

    Video analizinde her frame'i işlemek özellikle Faster R-CNN için çok yavaş
    olabilir. Bu nedenle `frame_stride` ile örnekleme yapılır. Örneğin
    frame_stride=5 ise yalnızca 0, 5, 10, 15... numaralı frame'ler analiz edilir.

    Args:
        video_path: Analiz edilecek video dosyasının yolu.
        detectors: Model adı -> detector nesnesi sözlüğü.
        conf_threshold: Model tahminlerinde kullanılacak confidence eşiği.
        target_classes: Trafik senaryosu için hedef sınıflar.
        max_frames: En fazla kaç örnek frame işleneceği.
        frame_stride: Frame örnekleme aralığı.

    Returns:
        Tuple[str, pd.DataFrame]:
            - Annotasyonlu karşılaştırma videosunun dosya yolu.
            - Frame bazlı metrikleri içeren DataFrame.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {video_path}")

    # Video metadata bilgileri okunur.
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video width/height could not be read correctly.")

    # Geçici çıktı dosyaları. raw_out_path OpenCV çıktısı, final_out_path H.264 çıktısıdır.
    timestamp = int(time.time())
    raw_out_path = os.path.join(tempfile.gettempdir(), f"od_comparison_raw_{timestamp}.mp4")
    final_out_path = os.path.join(tempfile.gettempdir(), f"od_comparison_h264_{timestamp}.mp4")

    # İki model seçildiyse çıktı videosu yan yana olacağı için genişlik model sayısıyla çarpılır.
    num_models = len(detectors)
    combined_width = width * num_models
    combined_height = height

    # Çok büyük videolar tarayıcıda oynatma sorununa neden olabilir.
    # Bu nedenle yan yana çıktı genişliği 1600 piksele ölçeklenir.
    max_output_width = 1600
    scale = min(1.0, max_output_width / combined_width)

    # H.264 encoder genellikle çift sayı genişlik/yükseklik ister.
    out_width = int(combined_width * scale)
    out_height = int(combined_height * scale)

    out_width = max(2, out_width - (out_width % 2))
    out_height = max(2, out_height - (out_height % 2))

    writer = cv2.VideoWriter(
        raw_out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, source_fps / max(1, frame_stride)),
        (out_width, out_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV VideoWriter could not be opened.")

    rows: List[dict] = []
    frame_id = 0
    processed = 0

    while processed < max_frames:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        # Frame örnekleme: seçili aralığa denk gelmeyen frame'ler atlanır.
        if frame_id % frame_stride != 0:
            frame_id += 1
            continue

        # OpenCV BGR frame'i PIL RGB formatına çevrilir.
        pil_image = bgr_to_pil(frame_bgr)
        annotated_frames = []

        for display_name, detector in detectors.items():
            # Her model için inference süresi ayrı ayrı ölçülür.
            start = time.perf_counter()
            detections = detector.predict(pil_image, conf_threshold, target_classes)
            elapsed = time.perf_counter() - start

            # Frame bazlı detection sayısı, küçük nesne sayısı, süre ve FPS hesaplanır.
            stats = aggregate_stats(detections, pil_image.size, elapsed)

            rows.append(
                {
                    "frame": frame_id,
                    "model": display_name,
                    **stats,
                }
            )

            # Aynı frame için model çıktısı çizilir.
            annotated_frames.append(
                draw_detections(pil_image, detections, display_name)
            )

        # Seçili modellerin çıktıları yan yana birleştirilir.
        combined = annotated_frames[0]

        for next_frame in annotated_frames[1:]:
            combined = concat_side_by_side(combined, next_frame)

        combined_bgr = pil_to_bgr(combined)

        # Çıktı videosunun boyutu VideoWriter ile birebir aynı olmalıdır.
        if combined_bgr.shape[1] != out_width or combined_bgr.shape[0] != out_height:
            combined_bgr = cv2.resize(
                combined_bgr,
                (out_width, out_height),
                interpolation=cv2.INTER_AREA,
            )

        writer.write(combined_bgr)

        processed += 1
        frame_id += 1

    cap.release()
    writer.release()

    if processed == 0:
        raise RuntimeError("No frames were processed from the video.")

    # OpenCV çıktısı browser uyumlu H.264 MP4 formatına dönüştürülür.
    output_path = _convert_to_browser_compatible_mp4(raw_out_path, final_out_path)

    return output_path, pd.DataFrame(rows)