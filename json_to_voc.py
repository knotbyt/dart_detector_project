"""Convert labelme / AnyLabeling / X-AnyLabeling JSON annotations to Pascal VOC XML.

Reads every .json in --in-dir and writes a matching .xml in --out-dir.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET


def to_voc(json_path, out_path, image_root=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = int(data.get('imageWidth', 0))
    img_h = int(data.get('imageHeight', 0))
    img_name = os.path.basename(data.get('imagePath', ''))
    if not img_name:
        img_name = os.path.splitext(os.path.basename(json_path))[0] + '.jpg'
    img_path = os.path.join(image_root, img_name) if image_root else img_name

    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text = os.path.basename(os.path.dirname(img_path)) or 'images'
    ET.SubElement(ann, 'filename').text = img_name
    ET.SubElement(ann, 'path').text = os.path.abspath(img_path)
    src = ET.SubElement(ann, 'source')
    ET.SubElement(src, 'database').text = 'Unknown'
    size = ET.SubElement(ann, 'size')
    ET.SubElement(size, 'width').text = str(img_w)
    ET.SubElement(size, 'height').text = str(img_h)
    ET.SubElement(size, 'depth').text = '3'
    ET.SubElement(ann, 'segmented').text = '0'

    shapes = data.get('shapes', [])
    n_objects = 0
    for shp in shapes:
        label = shp.get('label', '').strip()
        if not label:
            continue
        pts = shp.get('points', [])
        if not pts:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        xmin, xmax = max(0, min(xs)), min(img_w or max(xs), max(xs))
        ymin, ymax = max(0, min(ys)), min(img_h or max(ys), max(ys))
        if xmax <= xmin or ymax <= ymin:
            continue
        obj = ET.SubElement(ann, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bnd = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bnd, 'xmin').text = str(int(round(xmin)))
        ET.SubElement(bnd, 'ymin').text = str(int(round(ymin)))
        ET.SubElement(bnd, 'xmax').text = str(int(round(xmax)))
        ET.SubElement(bnd, 'ymax').text = str(int(round(ymax)))
        n_objects += 1

    tree = ET.ElementTree(ann)
    ET.indent(tree, space='  ')
    tree.write(out_path, encoding='utf-8', xml_declaration=False)
    return n_objects


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dir', default='data/annotations',
                   help="Folder with JSON annotations")
    p.add_argument('--out-dir', default='data/annotations',
                   help="Folder for output VOC XMLs (can be same as --in-dir)")
    p.add_argument('--images', default='data/images',
                   help="Folder containing source images (for <path> tag)")
    args = p.parse_args()

    if not os.path.isdir(args.in_dir):
        print(f"Not a folder: {args.in_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    jsons = [f for f in os.listdir(args.in_dir) if f.lower().endswith('.json')]
    print(f"Found {len(jsons)} JSON files in {args.in_dir}")

    total_objs = 0
    for fn in jsons:
        json_path = os.path.join(args.in_dir, fn)
        xml_path = os.path.join(args.out_dir, os.path.splitext(fn)[0] + '.xml')
        try:
            n = to_voc(json_path, xml_path, image_root=args.images)
            total_objs += n
            print(f"  {fn} -> {os.path.basename(xml_path)} ({n} objects)")
        except Exception as e:
            print(f"  FAILED {fn}: {e}")

    print(f"\nDone. {len(jsons)} files, {total_objs} total boxes.")


if __name__ == '__main__':
    main()
