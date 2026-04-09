from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(repo_id="jamesnguyen831/fruit-detection-yolov8s", filename="best.pt")
dst = os.path.join("backend", "weights", "best.pt")
shutil.copy(path, dst)
print("Downloaded best.pt ->", dst)
sz = os.path.getsize(dst)
print(f"Size: {sz / 1024 / 1024:.1f} MB")
