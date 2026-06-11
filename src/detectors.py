"""
YOLOv8n ve Faster R-CNN model sarmalayıcıları.

Bu dosyanın temel amacı, iki farklı kütüphaneden gelen model çıktılarını ortak
bir formata dönüştürmektir. YOLOv8n Ultralytics üzerinden, Faster R-CNN ise
TorchVision üzerinden çalıştırılır. İki modelin ham çıktıları farklı yapıda
olduğu için proje içinde ortak bir detection şeması kullanılır:

    {
        "model": str,
        "class_name": str,
        "score": float,
        "box": [x1, y1, x2, y2]
    }

Bu ortak şema sayesinde evaluation.py ve visualization.py modellerden bağımsız
çalışabilir.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class Detection:
    """
    Tek bir nesne tespitini temsil eden veri sınıfı.

    Attributes:
        model: Tahmini üreten model adı.
        class_name: Tahmin edilen sınıf adı. Örnek: car, person, traffic light.
        score: Modelin confidence skoru.
        box: [x1, y1, x2, y2] formatında bounding box koordinatları.
    """

    model: str
    class_name: str
    score: float
    box: list[float]

    def to_dict(self) -> dict:
        """Detection nesnesini proje genelinde kullanılan sözlük formatına çevirir."""
        return {
            "model": self.model,
            "class_name": self.class_name,
            "score": float(self.score),
            "box": [float(x) for x in self.box],
        }


class BaseDetector:
    """
    Tüm model sarmalayıcıları için ortak arayüz.

    YOLOv8Detector ve FasterRCNNDetector bu sınıfın predict imzasını takip eder.
    Böylece app.py içinde model türüne göre özel kod yazmaya gerek kalmaz.
    """

    model_name: str

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.30,
        target_classes: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """
        Verilen görüntü üzerinde detection çalıştırır.

        Alt sınıflar bu metodu override eder.
        """
        raise NotImplementedError


class YOLOv8Detector(BaseDetector):
    """Ultralytics YOLOv8n model sarmalayıcısı (wrapper)."""

    def __init__(self, weights: str = "yolov8n.pt", device: str = "auto") -> None:
        # Ultralytics importu burada yapılır; böylece modül import edildiğinde değil,
        # model gerçekten kullanılacağı zaman kütüphane yüklenir.
        from ultralytics import YOLO

        self.model_name = "YOLOv8n"
        self.device = _resolve_device(device)

        # weights='yolov8n.pt' ilk çalıştırmada Ultralytics tarafından indirilebilir.
        self.model = YOLO(weights)

    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.30,
        target_classes: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """YOLOv8n modeli ile tahmin üretir ve ortak detection formatına dönüştürür."""
        target_set = set(target_classes or [])

        # Ultralytics modeli numpy RGB array ile çalışabilir.
        np_image = np.asarray(image.convert("RGB"))

        # verbose=False terminal çıktısını azaltır.
        results = self.model.predict(
            source=np_image,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return []

        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = names.get(cls_id, str(cls_id))

            # Kullanıcı hedef sınıf seçmişse sadece bu sınıflar tutulur.
            if target_set and class_name not in target_set:
                continue
            score = float(box.conf[0].item())
            xyxy = box.xyxy[0].detach().cpu().tolist()
            detections.append(Detection(self.model_name, class_name, score, xyxy))
        return [d.to_dict() for d in detections]


class FasterRCNNDetector(BaseDetector):
    """TorchVision Faster R-CNN ResNet50-FPN model sarmalayıcısı (wrapper)."""

    def __init__(self, device: str = "auto") -> None:
        # TorchVision importları sadece model kullanılacağı zaman yapılır.
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
        from torchvision.transforms.functional import to_tensor

        self.model_name = "Faster R-CNN"
        self.device = torch.device(_resolve_device(device))

        # COCO üzerinde önceden eğitilmiş ağırlıklar kullanılır.
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]

        # COCO sınıf isimleri metadata içinden alınır.
        self.to_tensor = to_tensor
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(self.device)

        # Model inference moduna alınır. Dropout/BN gibi eğitim davranışları devre dışı kalır.
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.30,
        target_classes: Optional[Iterable[str]] = None,
    ) -> List[dict]:
        """Faster R-CNN modeli ile tahmin üretir ve ortak detection formatına dönüştürür."""
        target_set = set(target_classes or [])

        # TorchVision detection modelleri [C, H, W] formatında torch tensor bekler.
        tensor = self.to_tensor(image.convert("RGB")).to(self.device)

        # Tek görüntü verildiği için çıktı listesinin ilk elemanı alınır.
        outputs = self.model([tensor])[0]

        detections: List[Detection] = []
        for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
            score_value = float(score.detach().cpu().item())

            # Confidence threshold altında kalan tahminler rapora dahil edilmez.
            if score_value < conf_threshold:
                continue
            label_id = int(label.detach().cpu().item())
            class_name = self.categories[label_id] if label_id < len(self.categories) else str(label_id)
            if target_set and class_name not in target_set:
                continue
            detections.append(
                Detection(
                    self.model_name,
                    class_name,
                    score_value,
                    box.detach().cpu().tolist(),
                )
            )
        return [d.to_dict() for d in detections]


def _resolve_device(device: str) -> str:
    """
    Kullanıcı cihaz tercihini gerçek çalıştırma cihazına dönüştürür.

    Args:
        device: "auto", "cuda" veya "cpu".

    Returns:
        PyTorch ve Ultralytics tarafından kabul edilen cihaz adı.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    # Kullanıcı CUDA seçmiş ama sistemde CUDA yoksa uygulama çökmesin; CPU'ya düşsün.
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device
