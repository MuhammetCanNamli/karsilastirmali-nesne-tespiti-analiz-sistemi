"""Klasördeki tüm görüntüler üzerinde toplu model analizi yapar.

Bu script Streamlit arayüzünden bağımsızdır. Bir klasör içindeki tüm görüntüler
üzerinde YOLOv8n ve Faster R-CNN modellerini sırayla çalıştırır ve CSV çıktıları
üretir. Raporun deneysel sonuç tablolarını doldurmak için kullanışlıdır.

Örnek kullanım:
    python scripts/run_image_batch.py --input data/samples --output outputs/batch_results.csv

Üretilen çıktılar:
    - outputs/batch_results.csv: Görüntü/model bazlı özet metrikler
    - outputs/<image>_<model>_detections.csv: Tek tek detection detayları
"""


from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

# Script scripts/ klasörü altında çalıştığı için proje kök dizini sys.path'e eklenir.
# Bu sayede src modülleri import edilebilir.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DEFAULT_CONFIDENCE, TRAFFIC_CLASSES  # noqa: E402
from src.detectors import FasterRCNNDetector, YOLOv8Detector  # noqa: E402
from src.evaluation import aggregate_stats, summarize_detections  # noqa: E402


def main() -> None:
    """Komut satırı argümanlarını okur, modelleri çalıştırır ve CSV üretir."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image folder")
    parser.add_argument("--output", default="outputs/batch_results.csv")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Her iki model de bir kez yüklenir ve tüm görüntüler için tekrar kullanılır.
    detectors = {
        "YOLOv8n": YOLOv8Detector(device=args.device),
        "Faster R-CNN": FasterRCNNDetector(device=args.device),
    }

    rows = []

    # Klasördeki desteklenen görüntü dosyaları sırayla işlenir.
    for image_path in sorted(input_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        
        image = Image.open(image_path).convert("RGB")

        for model_name, detector in detectors.items():
            # Inference süresi ölçülür.
            start = time.perf_counter()
            detections = detector.predict(image, args.conf, TRAFFIC_CLASSES)
            elapsed = time.perf_counter() - start

            # Özet istatistikler ana batch tablosuna eklenir.
            stats = aggregate_stats(detections, image.size, elapsed)
            rows.append({"image": image_path.name, "model": model_name, **stats})

            # Tek tek detection detayları ayrı CSV olarak kaydedilir.
            detail_df = summarize_detections(detections, image.size)
            detail_out = output_path.parent / f"{image_path.stem}_{model_name.replace(' ', '_')}_detections.csv"
            detail_df.to_csv(detail_out, index=False)

    # Tüm görüntü/model özetleri tek CSV olarak kaydedilir.
    pd.DataFrame(rows).to_csv(output_path, index=False)
    
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
