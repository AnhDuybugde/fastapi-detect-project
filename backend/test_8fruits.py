"""Test all 8 fruit classes with YOLO-World via API."""
import requests, os, sys
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_images")

test_cases = [
    ("test_banana.jpg",     "banana"),
    ("test_apple.jpg",      "apple"),
    ("test_orange.png",     "orange"),
    ("test_grape.png",      "grape"),
    ("test_watermelon.png", "watermelon"),
    ("test_strawberry.png", "strawberry"),
    ("test_mango.png",      "mango"),
    ("test_pineapple.png",  "pineapple"),
]

print("=" * 60)
print("  YOLO-World — 8 Fruit Classes Full Test")
print("=" * 60)

passed = 0
failed = 0

for filename, expected in test_cases:
    path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(path):
        print(f"\n❌ [SKIP] {filename} not found")
        failed += 1
        continue

    ct = "image/jpeg" if filename.endswith(".jpg") else "image/png"
    with open(path, "rb") as f:
        r = requests.post(
            f"{API}/api/detect-upload",
            files={"image": (filename, f, ct)},
            data={"confidence_threshold": "0.50"}, # Đã ứng dụng Strategy 2
        )

    if r.status_code != 200:
        print(f"\n❌ [{expected.upper()}] API Error {r.status_code}")
        failed += 1
        continue

    data = r.json()
    classes_found = [d["class_name"].lower() for d in data["detections"]]
    top_class = classes_found[0] if classes_found else "nothing"
    top_conf = data["detections"][0]["confidence"] if data["detections"] else 0

    ok = expected in classes_found
    icon = "✅" if ok else "❌"
    status = "OK" if ok else "FAIL"

    if ok:
        passed += 1
    else:
        failed += 1

    print(f"\n{icon} [{expected.upper()}] {status}")
    print(f"   File: {filename}")
    print(f"   Detected: {classes_found} (top: {top_class} @ {top_conf:.1%})")
    if len(data["detections"]) > 1:
        for d in data["detections"]:
            print(f"     -> {d['class_name']} | {d['confidence']:.1%}")

print("\n" + "=" * 60)
print(f"  RESULT: {passed}/8 PASSED | {failed}/8 FAILED")
print("=" * 60)
