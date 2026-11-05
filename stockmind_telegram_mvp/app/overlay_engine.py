from io import BytesIO
from PIL import Image, ImageDraw


def render_overlay(image_bytes: bytes, zones: list[dict]) -> tuple[bytes | None, int]:
    """Draw validated, normalized rectangles onto a copy of the uploaded chart."""
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    count = 0
    for zone in zones[:8]:
        if not isinstance(zone, dict) or zone.get("draw_overlay") is not True:
            continue
        if zone.get("zone_type") not in {"supply", "demand"}:
            continue
        try:
            x0, x1, y0, y1 = [float(zone[k]) for k in ("x_start", "x_end", "y_top", "y_bottom")]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(0 <= v <= 1 for v in (x0, x1, y0, y1)):
            continue
        if x1 <= x0 or y1 <= y0 or y1 - y0 > 0.45:
            continue
        color = (220, 65, 65, 60) if zone["zone_type"] == "supply" else (30, 170, 100, 60)
        edge = (235, 65, 65, 230) if zone["zone_type"] == "supply" else (25, 185, 110, 230)
        box = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
        draw.rectangle(box, fill=color, outline=edge, width=max(2, width // 600))
        title = "SUPPLY" if zone["zone_type"] == "supply" else "DEMAND"
        label = str(zone.get("label") or title)[:36]
        draw.text((box[0] + 5, max(0, box[1] - 18)), label, fill=edge[:3] + (255,))
        count += 1
    if not count:
        return None, 0
    result = Image.alpha_composite(image, layer).convert("RGB")
    out = BytesIO()
    result.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue(), count
