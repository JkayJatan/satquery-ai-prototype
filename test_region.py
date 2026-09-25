from PIL import Image
from modules.region_detector import detect_regions

img = Image.open("sample_data/sample_before.png")
r = detect_regions(img, query="detect buildings and storage tanks", min_area=400, max_regions=8)
print("Regions detected:", r["num_regions"])
print("Mean saliency:", r["mean_saliency"])
for b in r["bounding_boxes"]:
    print(f"  R{b['id']}: x={b['x']} y={b['y']} w={b['w']} h={b['h']} saliency={b['saliency_score']}")
print("Region detection OK")
