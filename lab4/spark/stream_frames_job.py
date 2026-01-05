import os
import base64
import shutil
from pathlib import Path

from pyspark import SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from pyspark.sql.functions import col, from_json


def get_root():
    return Path(__file__).resolve().parents[1]


def ensure_model_local(basename):
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    src = Path(SparkFiles.get(basename))
    dst = models_dir / "selfie_segmenter.tflite"
    if not dst.exists():
        shutil.copy(src, dst)


def process_partition(rows, model_basename, output_dir):
    import cv2
    import numpy as np

    ensure_model_local(model_basename)
    import background_remover

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ok_cnt = 0
    err_cnt = 0

    for r in rows:
        try:
            cam_id = r["camera_id"]
            frame_id = r["frame_id"]
            jpg_b64 = r["jpg_b64"]

            jpg_bytes = base64.b64decode(jpg_b64.encode("utf-8"))
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("decode_none")

            out = background_remover.remove_background(frame)

            out_path = os.path.join(output_dir, f"{cam_id}_{frame_id}.jpg")
            if not cv2.imwrite(out_path, out):
                raise ValueError("write_fail")

            ok_cnt += 1
        except Exception:
            err_cnt += 1

    yield (ok_cnt, err_cnt)


def main():
    cam_host = os.getenv("CAM_HOST", os.getenv("CAM_STREAM_HOST", "localhost"))
    cam_port = int(os.getenv("CAM_PORT", os.getenv("CAM_STREAM_PORT", "6100")))

    root = get_root()
    model_path = os.getenv("MODEL_PATH", str(root / "models" / "selfie_segmenter.tflite"))
    output_dir = os.getenv("OUTPUT_DIR", str(root / "output"))
    ckpt_dir = os.getenv("CHECKPOINT_DIR", "/tmp/removebg_ckpt")

    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)

    bg_py = root / "background_remover.py"
    if not bg_py.exists():
        raise FileNotFoundError(str(bg_py))

    spark = (
        SparkSession.builder
        .appName("removebg_stream")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    spark.sparkContext.addPyFile(str(bg_py))
    spark.sparkContext.addFile(model_path)
    model_basename = Path(model_path).name

    schema = StructType([
        StructField("camera_id", StringType(), True),
        StructField("frame_id", LongType(), True),
        StructField("ts", DoubleType(), True),
        StructField("jpg_b64", StringType(), True),
    ])

    lines = (
        spark.readStr
