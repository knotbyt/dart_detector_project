# Darts Score Detection

## Install

```
pip install opencv-python onnxruntime numpy
```

## Find your camera index

```
python -c "import cv2; [print(i, cv2.VideoCapture(i, cv2.CAP_DSHOW).read()[0]) for i in range(5)]"
```

Use the index that prints `True`.

## Run

```
python darts_score_detection_offline.py --input 0
```

Replace `0` with your camera index. Press `q` to quit.

### With capture mode (SPACE to save raw frames)

```
python darts_score_detection_offline.py --input 0 --capture
```

Saves frames to `data/images/`.

## Annotate captured images

Install labelme (works on Python 3.10+):

```
pip install labelme
```

Run it pointed at your captures folder:

```
labelme data/images --labels data/predefined_classes.txt --output data/annotations --nodata --autosave
```

For each image:
- Press `Ctrl+R` (Create Rectangle) → draw a box around the **Bull** → label `Bull`
- `Ctrl+R` → draw a box around the **arrow** → label with where it landed (`S20`, `T12`, `D5`, etc.)
- `D` to next image (auto-saves)

Annotations save as `.json` in `data/annotations/`.

## Convert annotations to Pascal VOC XML

The training scripts read VOC XML, not JSON. Run:

```
python json_to_voc.py
```

This reads every `.json` in `data/annotations/` and writes a matching `.xml` next to it.

## Train the area-discrimination DNN

```
python feature_creation.py --annotations-dir data/annotations/ --output-file data/features.tsv
python score_detection_training.py --input-data data/features.tsv --output-dir models/dnn/ --split-percent 0.6 --train-epochs 500 --onnx-option True
```

Note the **trailing slash** on `--annotations-dir data/annotations/` — required.
