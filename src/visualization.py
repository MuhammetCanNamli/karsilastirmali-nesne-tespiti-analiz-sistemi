"""
Detection sonuçlarını görselleştirme yardımcıları.

Bu modül, model tahminlerini rapor ve Streamlit arayüzünde anlaşılır hale
getirmek için kullanılır. Temel görevler:

- Görüntü üzerine bounding box çizmek
- Sınıf adı ve confidence değerini yazmak
- İki model çıktısını yan yana birleştirmek
- OpenCV BGR formatı ile PIL RGB formatı arasında dönüşüm yapmak
"""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Farklı detection kutularını ayırt etmek için kullanılan renk paleti.
# Renkler RGB formatındadır.
PALETTE = [
    (220, 20, 60),
    (30, 144, 255),
    (34, 139, 34),
    (255, 140, 0),
    (138, 43, 226),
    (0, 128, 128),
    (199, 21, 133),
    (80, 80, 80),
]


def draw_detections(image: Image.Image, detections: List[dict], title: str = "") -> Image.Image:
    """
    Görüntü üzerine detection kutularını ve etiketleri çizer.

    Args:
        image: PIL RGB görüntüsü.
        detections: Ortak detection sözlüklerinden oluşan liste.
        title: Görüntünün üstüne yazılacak model başlığı.

    Returns:
        Bounding box çizilmiş yeni PIL görüntüsü.
    """
    # Orijinal görüntü değişmesin diye copy alınır.
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    
    # DejaVuSans çoğu Linux/Windows ortamında bulunur. Bulunamazsa default font kullanılır.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, det in enumerate(detections):
        color = PALETTE[idx % len(PALETTE)]
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        label = f"{det['class_name']} {det['score']:.2f}"

        # Nesne kutusu çizilir.
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Etiketin arka plan kutusu hesaplanır.
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Etiket kutusu nesnenin üstüne çizilir. y1 çok yukarıdaysa görüntü dışına taşmaması sağlanır.
        draw.rectangle([x1, max(0, y1 - text_h - 6), x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - text_h - 4)), label, fill=(255, 255, 255), font=font)

    # Model adı gibi başlıklar için görüntünün üstüne koyu şerit eklenir.
    if title:
        draw.rectangle([0, 0, img.width, 32], fill=(0, 0, 0))
        draw.text((8, 7), title, fill=(255, 255, 255), font=font)
    return img


def concat_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """
    İki görüntüyü yatay olarak yan yana birleştirir.

    Model karşılaştırmasında YOLOv8n ve Faster R-CNN çıktılarının aynı satırda
    gösterilmesi için kullanılır.
    """
    left = left.convert("RGB")
    right = right.convert("RGB")
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """PIL RGB görüntüsünü OpenCV BGR numpy array formatına dönüştürür."""
    rgb = np.asarray(image.convert("RGB"))

    # OpenCV video writer BGR beklediği için RGB kanal sırası ters çevrilir.
    return rgb[:, :, ::-1].copy()


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV BGR frame'ini PIL RGB görüntüsüne dönüştürür."""
    return Image.fromarray(frame[:, :, ::-1])
