"""Capture still images from a webcam for annotation.

Usage:
  python capture_images.py --camera 1 --out data/images
Keys: SPACE = save frame, q = quit.
"""
import argparse
import os
import time

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--out', default='data/images')
    p.add_argument('--prefix', default='darts')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        cv2.putText(preview, f"SPACE=save  q=quit  saved={saved}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('capture', preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == 32:  # SPACE
            name = f"{args.prefix}_{int(time.time()*1000)}.jpg"
            cv2.imwrite(os.path.join(args.out, name), frame)
            saved += 1
            print(f"saved {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
