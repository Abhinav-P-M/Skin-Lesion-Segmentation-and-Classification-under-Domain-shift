"""
Pretrained MobileNetV2 Skin Cancer Classifier Wrapper.
Loads the pretrained Keras .h5 model and exposes a unified interface
compatible with the rest of the PyTorch-based pipeline.
"""

import os
import numpy as np
from PIL import Image

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# Path to the pretrained model
H5_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "pretrained", "full_skin_cancer_model.h5"
)

HAM10000_CLASSES = [
    "akiec",  # 0 - Actinic Keratoses
    "bcc",    # 1 - Basal Cell Carcinoma
    "bkl",    # 2 - Benign Keratosis
    "df",     # 3 - Dermatofibroma
    "mel",    # 4 - Melanoma
    "nv",     # 5 - Melanocytic Nevi
    "vasc",   # 6 - Vascular Lesions
]

CLASS_NAMES_FULL = [
    "Actinic Keratoses & Intraepithelial Carcinoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions",
]

MALIGNANT_CLASSES = {0, 1, 4}  # akiec, bcc, mel


class PretrainedSkinCancerClassifier:
    """
    Wrapper around the pretrained MobileNetV2 Keras model.
    Input : PIL Image or numpy array (any size, auto-resized to 224x224)
    Output: dict with probabilities, predicted class, confidence
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or H5_MODEL_PATH
        self._model = None

    def _load(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Pretrained model not found at: {self.model_path}\n"
                    f"Copy full_skin_cancer_model.h5 to models/pretrained/"
                )
            self._model = tf.keras.models.load_model(self.model_path, compile=False)

    def preprocess(self, image) -> np.ndarray:
        """Resize to 224x224, normalize to [0,1], add batch dim."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
            image = np.array(image, dtype=np.float32) / 255.0
        # Expect (224, 224, 3)
        return np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

    def predict(self, image) -> dict:
        """
        Run inference on a single image.
        Returns:
            {
              'probs': list[float] (7 class probabilities),
              'pred_idx': int,
              'pred_class': str,
              'pred_name': str,
              'confidence': float,
              'is_malignant': bool
            }
        """
        self._load()
        x = self.preprocess(image)
        probs = self._model.predict(x, verbose=0)[0]  # shape (7,)
        pred_idx = int(np.argmax(probs))
        return {
            "probs": probs.tolist(),
            "pred_idx": pred_idx,
            "pred_class": HAM10000_CLASSES[pred_idx],
            "pred_name": CLASS_NAMES_FULL[pred_idx],
            "confidence": float(probs[pred_idx]),
            "is_malignant": pred_idx in MALIGNANT_CLASSES,
        }

    def predict_batch(self, images: list) -> list:
        """Run inference on a list of PIL images or numpy arrays."""
        self._load()
        batch = np.concatenate([self.preprocess(img) for img in images], axis=0)
        probs_batch = self._model.predict(batch, verbose=0)
        results = []
        for probs in probs_batch:
            pred_idx = int(np.argmax(probs))
            results.append({
                "probs": probs.tolist(),
                "pred_idx": pred_idx,
                "pred_class": HAM10000_CLASSES[pred_idx],
                "pred_name": CLASS_NAMES_FULL[pred_idx],
                "confidence": float(probs[pred_idx]),
                "is_malignant": pred_idx in MALIGNANT_CLASSES,
            })
        return results

    def get_feature_vector(self, image) -> np.ndarray:
        """Extract 1280-dim MobileNetV2 feature embedding (before classifier head)."""
        self._load()
        # Build feature extractor lazily
        if not hasattr(self, "_feat_model"):
            # The inner MobileNetV2 submodel output before Dense head
            self._feat_model = tf.keras.Model(
                inputs=self._model.input,
                outputs=self._model.layers[-3].output  # after Flatten, before Dense
            )
        x = self.preprocess(image)
        return self._feat_model.predict(x, verbose=0)[0]


# Module-level singleton (lazy-loaded on first use)
_classifier = None


def get_classifier(model_path: str = None) -> PretrainedSkinCancerClassifier:
    """Return cached singleton classifier."""
    global _classifier
    if _classifier is None:
        _classifier = PretrainedSkinCancerClassifier(model_path)
    return _classifier
