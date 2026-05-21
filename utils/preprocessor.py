from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


class Preprocessor:
    """OCR 품질 향상을 위한 이미지 전처리."""

    UPSCALE_FACTOR = 2
    LOW_DPI_THRESHOLD = 150

    def process(self, image: Image.Image, is_low_res: bool = False) -> Image.Image:
        """전처리 파이프라인 전체를 실행한다."""
        img = self._to_cv2(image)

        if is_low_res:
            img = self._upscale(img)

        img = self._deskew(img)
        img = self._binarize(img)

        return self._to_pil(img)

    # ------------------------------------------------------------------
    # 개별 처리 단계
    # ------------------------------------------------------------------

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """저해상도 이미지를 2배 확대한다."""
        h, w = img.shape[:2]
        return cv2.resize(
            img,
            (w * self.UPSCALE_FACTOR, h * self.UPSCALE_FACTOR),
            interpolation=cv2.INTER_CUBIC,
        )

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """HoughLines 기반 기울기 감지 후 보정한다."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

        if lines is None:
            return img

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = (theta - np.pi / 2) * (180 / np.pi)
            if abs(angle) < 10:
                angles.append(angle)

        if not angles:
            return img

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.5:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Adaptive thresholding으로 이진화한다."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        return binary

    # ------------------------------------------------------------------
    # 변환 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cv2(image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _to_pil(img: np.ndarray) -> Image.Image:
        if len(img.shape) == 2:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
