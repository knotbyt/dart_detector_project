"""Patch labelImg's canvas.py to cast float coords to int so it runs on PyQt5 >= 5.15 / Python 3.10+."""
import os, re, sys
import labelImg

root = os.path.dirname(labelImg.__file__)
targets = [
    os.path.join(root, 'libs', 'canvas.py'),
    os.path.join(root, 'libs', 'shape.py'),
]
# Patterns: wrap bare .x() / .y() calls inside draw* / QPoint* / QRect* with int()
patterns = [
    (re.compile(r'(p\.draw\w+\s*\([^)]*?)(self\.\w+_point\.x\(\))'), r'\1int(\2)'),
    (re.compile(r'(p\.draw\w+\s*\([^)]*?)(self\.\w+_point\.y\(\))'), r'\1int(\2)'),
    (re.compile(r'(p\.draw\w+\s*\([^)]*?)(self\.pixmap\.width\(\))'), r'\1int(\2)'),
    (re.compile(r'(p\.draw\w+\s*\([^)]*?)(self\.pixmap\.height\(\))'), r'\1int(\2)'),
]
for fn in targets:
    if not os.path.isfile(fn):
        print("skip (not found):", fn); continue
    with open(fn, 'r', encoding='utf-8') as f:
        src = f.read()
    orig = src
    for pat, repl in patterns:
        while True:
            new = pat.sub(repl, src)
            if new == src:
                break
            src = new
    if src != orig:
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(src)
        print("patched:", fn)
    else:
        print("no changes needed:", fn)
print("done.")
