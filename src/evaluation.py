"""
Detection değerlendirme ve model karşılaştırma yardımcıları.

Bu modül, model çıktılarının sayısal olarak analiz edilmesini sağlar:

- Bounding box alanı hesaplama
- IoU hesaplama
- Küçük nesne kontrolü
- Detection özet tablosu üretme
- İki modeli IoU tabanlı karşılaştırma
- Ground-truth verisi varsa TP / FP / FN analizi yapma
"""


from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import COCO_SMALL_AREA_PX, SMALL_OBJECT_AREA_RATIO


def box_area(box: Iterable[float]) -> float:
    """
    Bir bounding box'ın piksel alanını hesaplar.

    Args:
        box: [x1, y1, x2, y2] formatında kutu koordinatları.

    Returns:
        Kutunun alanı. Geçersiz kutularda negatif alan yerine 0 döndürülür.
    """
    x1, y1, x2, y2 = box

    # max(0, ...) kullanımı ters/bozuk koordinatlarda negatif alan oluşmasını engeller.
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """
    İki bounding box arasındaki IoU değerini hesaplar.

    IoU = intersection_area / union_area

    IoU değeri 0 ile 1 arasındadır:
        - 0: Kutular hiç örtüşmüyor.
        - 1: Kutular tamamen aynı.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Kesişim dikdörtgeninin koordinatları.
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter = box_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union = box_area(box_a) + box_area(box_b) - inter

    # union sıfır veya negatifse kutular geçersizdir; güvenli olarak 0 döndürülür.
    return 0.0 if union <= 0 else inter / union


def is_small_object(box: Iterable[float], image_size: Tuple[int, int]) -> bool:
    """
    Bir nesnenin küçük nesne kabul edilip edilmeyeceğini belirler.

    Bu projede iki pratik eşik birlikte kullanılır:
        1. Kutunun alanı 32x32 pikselden küçükse küçük nesne sayılır.
        2. Kutunun alanı toplam görüntü alanının %1'inden küçükse küçük nesne sayılır.

    Bu yaklaşım, uzak trafik ışığı veya uzaktaki araç gibi küçük nesneleri
    raporda sayısal olarak ayırmaya yarar.
    """
    width, height = image_size
    area = box_area(box)
    area_ratio = area / float(width * height)
    return area <= COCO_SMALL_AREA_PX or area_ratio <= SMALL_OBJECT_AREA_RATIO


def summarize_detections(detections: List[dict], image_size: Tuple[int, int]) -> pd.DataFrame:
    """
    Tek tek detection sonuçlarını tabloya dönüştürür.

    Streamlit arayüzünde her modelin tahminlerini satır satır göstermek için kullanılır.
    """
    rows = []
    for d in detections:
        rows.append(
            {
                "model": d["model"],
                "class": d["class_name"],
                "score": round(float(d["score"]), 3),
                "area_px": round(box_area(d["box"]), 1),
                "small_object": is_small_object(d["box"], image_size),
                "box": [round(x, 1) for x in d["box"]],
            }
        )
    return pd.DataFrame(rows)


def aggregate_stats(detections: List[dict], image_size: Tuple[int, int], elapsed_s: float) -> dict:
    """
    Bir modelin tek görüntü/frame üzerindeki özet metriklerini hesaplar.

    Hesaplanan değerler rapordaki ana performans tablolarına doğrudan yazılabilir:
        - detections
        - avg_confidence
        - small_objects
        - runtime_s
        - fps_estimate
    """
    if not detections:
        return {
            "detections": 0,
            "avg_confidence": 0.0,
            "small_objects": 0,
            "runtime_s": round(elapsed_s, 4),
            "fps_estimate": round(1.0 / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        }
    small_count = sum(is_small_object(d["box"], image_size) for d in detections)
    return {
        "detections": len(detections),
        "avg_confidence": round(float(np.mean([d["score"] for d in detections])), 3),
        "small_objects": int(small_count),
        "runtime_s": round(elapsed_s, 4),
        "fps_estimate": round(1.0 / elapsed_s, 2) if elapsed_s > 0 else 0.0,
    }


def compare_two_models(
    detections_a: List[dict],
    detections_b: List[dict],
    iou_threshold: float = 0.50,
) -> pd.DataFrame:
    """İki modelin çıktılarını aynı sınıf + IoU mantığıyla karşılaştırır.

    Bu fonksiyon ground-truth yerine geçmez. Burada amaç, iki modelin aynı
    nesnelerde ne kadar anlaştığını ölçmektir.

    Durumlar:
        matched:
            İki model aynı sınıfı, IoU eşiğini aşacak şekilde tespit etmiştir.
        model_a_only:
            Nesne yalnızca ilk model tarafından bulunmuştur.
            Bu durum model A için FP veya model B için FN adayıdır.
        model_b_only:
            Nesne yalnızca ikinci model tarafından bulunmuştur.
            Bu durum model B için FP veya model A için FN adayıdır.
    """
    used_b = set()
    rows = []

    # Model A'daki her detection için model B'deki en iyi eşleşme aranır.
    for i, da in enumerate(detections_a):
        best_j = None
        best_iou = 0.0

        for j, db in enumerate(detections_b):
            # Aynı model B kutusu birden fazla model A kutusuyla eşleştirilmesin.
            # Farklı sınıflar aynı nesne olarak kabul edilmez.
            if j in used_b or da["class_name"] != db["class_name"]:
                continue
            iou = bbox_iou(da["box"], db["box"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou >= iou_threshold:
            used_b.add(best_j)
            db = detections_b[best_j]
            rows.append(
                {
                    "class": da["class_name"],
                    "status": "matched",
                    "iou": round(best_iou, 3),
                    "model_a_score": round(float(da["score"]), 3),
                    "model_b_score": round(float(db["score"]), 3),
                    "interpretation": "Both models detect the same object.",
                }
            )
        else:
            rows.append(
                {
                    "class": da["class_name"],
                    "status": "model_a_only",
                    "iou": round(best_iou, 3),
                    "model_a_score": round(float(da["score"]), 3),
                    "model_b_score": None,
                    "interpretation": "Candidate FP for model A or candidate FN for model B.",
                }
            )

    # Model B'de olup model A ile eşleşmeyen kutular ayrıca raporlanır.
    for j, db in enumerate(detections_b):
        if j not in used_b:
            rows.append(
                {
                    "class": db["class_name"],
                    "status": "model_b_only",
                    "iou": 0.0,
                    "model_a_score": None,
                    "model_b_score": round(float(db["score"]), 3),
                    "interpretation": "Candidate FP for model B or candidate FN for model A.",
                }
            )
    return pd.DataFrame(rows)


def evaluate_against_ground_truth(
    predictions: List[dict],
    ground_truth: List[dict],
    iou_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Tahminleri ground-truth kutularına göre TP / FP / FN olarak değerlendirir.

    Ground-truth item formatı:
        {"class_name": "car", "box": [x1, y1, x2, y2]}

    Değerlendirme mantığı:
        - Aynı sınıfta ve IoU >= threshold olan en iyi eşleşme TP sayılır.
        - Eşleşemeyen tahminler FP sayılır.
        - Hiçbir tahminle eşleşmeyen ground-truth nesneleri FN sayılır.

    Not:
        Bu basit bir IoU tabanlı değerlendirmedir. COCO mAP hesabı değildir.
        Proje raporunda FP/FN örneklerini göstermek için yeterlidir.
    """
    matched_gt = set()
    rows = []
    for pred in predictions:
        best_i = None
        best_iou = 0.0

        # Tahmin için aynı sınıftaki en iyi ground-truth kutusu aranır.
        for i, gt in enumerate(ground_truth):
            if i in matched_gt or pred["class_name"] != gt["class_name"]:
                continue
            iou = bbox_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i is not None and best_iou >= iou_threshold:
            matched_gt.add(best_i)
            status = "TP"
        else:
            status = "FP"
        rows.append(
            {
                "model": pred["model"],
                "class": pred["class_name"],
                "status": status,
                "iou": round(best_iou, 3),
                "score": round(float(pred["score"]), 3),
            }
        )
    
    # Eşleşmeyen ground-truth nesneleri false negative olarak eklenir.
    for i, gt in enumerate(ground_truth):
        if i not in matched_gt:
            rows.append(
                {
                    "model": "-",
                    "class": gt["class_name"],
                    "status": "FN",
                    "iou": 0.0,
                    "score": None,
                }
            )
    return pd.DataFrame(rows)
