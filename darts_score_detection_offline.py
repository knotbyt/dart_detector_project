"""Offline port of darts_score_detection.py — runs on any machine with
Python + onnxruntime + opencv, no Jetson/CUDA required.

Example:
  python darts_score_detection_offline.py --input 0
  python darts_score_detection_offline.py --input path/to/video.mp4
  python darts_score_detection_offline.py --input path/to/image.jpg
  python darts_score_detection_offline.py --input 1 --capture
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime


SSD_INPUT_SIZE = 300
SSD_MEAN = np.array([127.0, 127.0, 127.0], dtype=np.float32)
SSD_SCALE = 1.0 / 128.0


def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, (SSD_INPUT_SIZE, SSD_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - SSD_MEAN) * SSD_SCALE
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]
    return img


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def postprocess(scores, boxes, img_w, img_h, conf_threshold):
    scores = scores[0]
    boxes = boxes[0]
    num_classes = scores.shape[1]
    results = []
    for class_id in range(1, num_classes):
        cls_scores = scores[:, class_id]
        mask = cls_scores >= conf_threshold
        if not np.any(mask):
            continue
        cls_boxes = boxes[mask]
        cls_scores_f = cls_scores[mask]
        keep = nms(cls_boxes, cls_scores_f)
        for k in keep:
            x1, y1, x2, y2 = cls_boxes[k]
            results.append((
                class_id,
                float(cls_scores_f[k]),
                float(x1 * img_w),
                float(y1 * img_h),
                float(x2 * img_w),
                float(y2 * img_h),
            ))
    return results


class DartsScoreDetection:
    def __init__(self, score_range, dnn_model):
        self.points = [11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 0]
        self.index_to_label = {0: 'Double', 1: 'Single', 2: 'Triple'}
        self.score_range = score_range
        self.dnn_model = dnn_model

    def calculate_score(self, x1, y1, x2, y2, box_width, box_height):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        rad = math.atan2(y2 - y1, x2 - x1)
        ort_inputs = {self.dnn_model.get_inputs()[0].name:
                      np.array([[distance, rad, box_width, box_height]], dtype=np.float32)}
        multiple = int(np.argmax(self.dnn_model.run(None, ort_inputs)[0]))
        res = self._binary_search(self.score_range, rad)
        predict_score = self.points[res]
        return f"{self.index_to_label[multiple]} {predict_score}"

    def _binary_search(self, numbers, value):
        left, right = 0, len(numbers) - 1
        while left <= right:
            mid = (left + right) // 2
            if mid + 1 < len(numbers) and numbers[mid] <= value < numbers[mid + 1]:
                return mid
            if numbers[mid] < value:
                left = mid + 1
            else:
                right = mid - 1
        return 21


def build_score_range():
    unit = np.pi / 10
    scale = -unit / 2
    radians = [scale]
    for _ in range(10):
        scale += unit
        radians.append(scale)
    radians.append(np.pi)
    return sorted([i * -1 for i in radians][2:]) + sorted(radians)


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='0',
                   help="Webcam index (e.g. '0') or path to video/image file")
    p.add_argument('--model', default='models/ssd/ssd-mobilenet.onnx')
    p.add_argument('--labels', default='models/ssd/new_labels.txt')
    p.add_argument('--dnn-model', default='models/dnn/dnn_model.onnx')
    p.add_argument('--threshold', type=float, default=0.3)
    p.add_argument('--output', default='',
                   help="Optional output video path (e.g. out.mp4)")
    p.add_argument('--capture', action='store_true',
                   help="Enable SPACE-to-save screenshots of the raw (un-annotated) frame")
    p.add_argument('--capture-dir', default='data/images',
                   help="Folder where screenshots are saved when --capture is set")
    args = p.parse_args()

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels: {labels}")

    ssd = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    dnn = onnxruntime.InferenceSession(args.dnn_model, providers=['CPUExecutionProvider'])
    ssd_input_name = ssd.get_inputs()[0].name

    dsd = DartsScoreDetection(score_range=build_score_range(), dnn_model=dnn)

    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    is_image = os.path.isfile(args.input) and args.input.lower().endswith(image_exts)

    if is_image:
        frames = [cv2.imread(args.input)]
        if frames[0] is None:
            print(f"Failed to read image: {args.input}", file=sys.stderr)
            sys.exit(1)
        cap = None
    else:
        src = int(args.input) if args.input.isdigit() else args.input
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Failed to open input: {args.input}", file=sys.stderr)
            sys.exit(1)
        frames = None

    writer = None
    center_x, center_y = 0, 0

    def process(frame):
        nonlocal center_x, center_y, writer
        h, w = frame.shape[:2]
        inp = preprocess(frame)
        scores, boxes = ssd.run(None, {ssd_input_name: inp})
        detections = postprocess(scores, boxes, w, h, args.threshold)

        bulls = [d for d in detections if d[0] == 1]
        if bulls:
            _, _, x1, y1, x2, y2 = bulls[0]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, 'Bull', (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for cid, conf, x1, y1, x2, y2 in detections:
            if cid == 1:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            label = dsd.calculate_score(center_x, center_y, cx, cy, bw, bh)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
            cv2.putText(frame, f"score: {label}", (int(x1) + 5, int(y1) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if args.output and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))
        if writer is not None:
            writer.write(frame)
        return frame

    if is_image:
        out = process(frames[0])
        cv2.imshow('Darts Score Detection', out)
        print("Press any key in the window to exit.")
        cv2.waitKey(0)
    else:
        if args.capture:
            os.makedirs(args.capture_dir, exist_ok=True)
            print(f"Capture mode on — press SPACE to save a raw frame to {args.capture_dir}")
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw = frame.copy()
            out = process(frame)
            if args.capture:
                cv2.putText(out, f"SPACE=save  q=quit  saved={saved}",
                            (10, out.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Darts Score Detection', out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if args.capture and k == 32:  # SPACE
                name = f"darts_{int(time.time() * 1000)}.jpg"
                path = os.path.join(args.capture_dir, name)
                cv2.imwrite(path, raw)
                saved += 1
                print(f"[capture #{saved}] saved {path}")
        cap.release()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
