import requests, sys
sys.stdout.reconfigure(encoding='utf-8')

# Test real watermelon photo
with open('Wassermelone_AH-1.jpg', 'rb') as f:
    r = requests.post(
        'http://127.0.0.1:8000/api/detect-upload',
        files={'image': ('watermelon.jpg', f, 'image/jpeg')},
        data={'confidence_threshold': '0.15'},
    )

d = r.json()
print('=== Real Watermelon Photo Test ===')
for x in d['detections'][:5]:
    name = x["class_name"]
    conf = x["confidence"]
    print(f'  {name} | {conf:.1%}')
if not d['detections']:
    print('  No detections')

# Test real images in root
import os
for img_name in ['flamegrapes2.jpg', 'chuoi-trai-01.jpg', 'pommier-oiase-apple-tree-f1.jpg']:
    if os.path.exists(img_name):
        with open(img_name, 'rb') as f:
            r = requests.post(
                'http://127.0.0.1:8000/api/detect-upload',
                files={'image': (img_name, f, 'image/jpeg')},
                data={'confidence_threshold': '0.15'},
            )
        d = r.json()
        print(f'\n=== {img_name} ===')
        for x in d['detections'][:3]:
            name = x["class_name"]
            conf = x["confidence"]
            print(f'  {name} | {conf:.1%}')
