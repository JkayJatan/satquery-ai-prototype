"""
Utility to generate synthetic bi-temporal satellite image pairs for testing SatQuery AI.
Generates:
- sample_data/sample_before.png (T0: Original landscape with river, vegetation, agricultural fields)
- sample_data/sample_after.png (T1: Landscape with new industrial facilities, cleared parcel, and access road)
"""

import os
from PIL import Image, ImageDraw, ImageFilter
import random


def generate_synthetic_satellite_pair(output_dir: str = "sample_data"):
    os.makedirs(output_dir, exist_ok=True)
    width, height = 512, 512
    random.seed(42)

    # 1. Base Terrain Canvas (T0 - Before)
    base_t0 = Image.new("RGB", (width, height), (170, 160, 130))
    draw_t0 = ImageDraw.Draw(base_t0)

    # Add background texture/terrain variance
    for _ in range(300):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(15, 60)
        shade = random.randint(-15, 15)
        color = (
            max(0, min(255, 170 + shade)),
            max(0, min(255, 160 + shade)),
            max(0, min(255, 130 + shade))
        )
        draw_t0.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # Agricultural plots / fields (T0)
    # Field 1 (Green crop)
    draw_t0.polygon([(40, 40), (180, 50), (170, 190), (30, 170)], fill=(75, 125, 60), outline=(50, 90, 40))
    # Field 2 (Golden harvest)
    draw_t0.polygon([(190, 50), (320, 60), (310, 180), (180, 180)], fill=(195, 165, 80), outline=(140, 115, 50))
    # Field 3 (Darker pasture)
    draw_t0.polygon([(40, 210), (170, 200), (180, 360), (50, 370)], fill=(60, 110, 50), outline=(40, 80, 35))
    # Field 4 (Fallow open field)
    draw_t0.polygon([(190, 200), (330, 210), (340, 350), (200, 360)], fill=(155, 140, 110), outline=(110, 95, 75))

    # Dense Forest patch in upper-right quadrant
    draw_t0.polygon([(360, 20), (490, 30), (500, 220), (350, 200)], fill=(35, 85, 40), outline=(25, 60, 30))
    for _ in range(40):
        fx = random.randint(360, 480)
        fy = random.randint(30, 200)
        draw_t0.ellipse([fx, fy, fx + 12, fy + 12], fill=(25, 70, 30))

    # Meandering River flowing through bottom half
    river_points = [
        (0, 430), (80, 420), (160, 440), (260, 425),
        (350, 450), (430, 435), (512, 460)
    ]
    draw_t0.line(river_points, fill=(45, 85, 130), width=32, joint="curve")
    draw_t0.line(river_points, fill=(65, 115, 170), width=18, joint="curve")

    # Primary transit road
    road_points = [(15, 0), (20, 512)]
    draw_t0.line(road_points, fill=(90, 90, 95), width=10)

    # Smooth base
    base_t0 = base_t0.filter(ImageFilter.GaussianBlur(1.2))

    # 2. Build T1 (After) by cloning T0 and introducing explicit physical changes
    base_t1 = base_t0.copy()
    draw_t1 = ImageDraw.Draw(base_t1)

    # CHANGE 1: New Industrial Complex in Field 4 (coordinates ~220, 230 to ~320, 330)
    # Concrete foundation pad
    draw_t1.rectangle([215, 225, 325, 335], fill=(130, 135, 140), outline=(80, 85, 90), width=2)
    # Large warehouse structure 1 (bright white-gray roof)
    draw_t1.rectangle([230, 240, 275, 320], fill=(235, 238, 242), outline=(50, 50, 60), width=2)
    # Large warehouse structure 2
    draw_t1.rectangle([285, 250, 315, 310], fill=(210, 215, 225), outline=(50, 50, 60), width=2)

    # CHANGE 2: Forest Clearing / Deforestation in upper-right forest
    # Cleared bare dirt zone
    draw_t1.ellipse([380, 60, 460, 150], fill=(195, 160, 115), outline=(150, 120, 80))

    # CHANGE 3: New Paved Access Road connecting main road to new warehouse
    draw_t1.line([(20, 275), (215, 275)], fill=(75, 75, 80), width=8)

    # CHANGE 4: Excavation / Retention Basin in lower terrain
    draw_t1.polygon([(360, 360), (430, 350), (440, 410), (370, 415)], fill=(30, 60, 95), outline=(20, 40, 70), width=2)

    base_t1 = base_t1.filter(ImageFilter.GaussianBlur(1.0))

    # Save to disk
    t0_path = os.path.join(output_dir, "sample_before.png")
    t1_path = os.path.join(output_dir, "sample_after.png")
    base_t0.save(t0_path)
    base_t1.save(t1_path)

    print(f"Generated sample satellite pair:\n- {t0_path}\n- {t1_path}")
    return t0_path, t1_path


if __name__ == "__main__":
    generate_synthetic_satellite_pair()
