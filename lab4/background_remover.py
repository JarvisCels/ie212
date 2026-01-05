import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_BG = (192, 192, 192)
_FG = (255, 255, 255)

_model = "models/selfie_segmenter.tflite"

_base = python.BaseOptions(model_asset_path=_model)
_cfg = vision.ImageSegmenterOptions(
    base_options=_base,
    output_category_mask=True
)
_segmenter = vision.ImageSegmenter.create_from_options(_cfg)


def remove_background(frame: np.ndarray) -> np.ndarray:
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    result = _segmenter.segment(mp_img)
    mask = result.category_mask.numpy_view()

    src = mp_img.numpy_view()

    fg = np.full(src.shape, _FG, dtype=np.uint8)
    bg = np.full(src.shape, _BG, dtype=np.uint8)

    selector = np.repeat(mask[:, :, None], 3, axis=2) > 0.2

    return np.where(selector, bg, src)
