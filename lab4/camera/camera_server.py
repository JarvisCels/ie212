import os
import json
import time
import base64
import socket
import glob
import cv2

HOST = os.getenv("CAM_HOST", "0.0.0.0")
PORT = int(os.getenv("CAM_PORT", "6100"))
VIDEO_PATH = os.getenv("VIDEO_PATH", "")
FPS = float(os.getenv("CAM_FPS", "2.0"))
JPEG_QUALITY = int(os.getenv("CAM_JPEG_QUALITY", "80"))

def pick_source():
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        return VIDEO_PATH
    vids = sorted(glob.glob(os.path.join("videos", "*.*")))
    return vids[0] if vids else ""

def frame_to_b64jpg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[camera] listening {HOST}:{PORT}")

    sleep_s = 1.0 / FPS if FPS > 0 else 0.5
    cam_id = os.getenv("CAM_ID", "cam01")

    while True:
        conn, addr = srv.accept()
        print(f"[camera] client {addr}")

        cap = None
        try:
            src = pick_source()
            cap = cv2.VideoCapture(src if src else 0)
            if not cap.isOpened():
                conn.close()
                continue

            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    if src:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                b64 = frame_to_b64jpg(frame)
                if b64 is None:
                    continue

                msg = {"camera_id": cam_id, "frame_id": frame_id, "ts": time.time(), "jpg_b64": b64}
                conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
                frame_id += 1
                time.sleep(sleep_s)

        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
