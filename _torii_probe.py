import base64, requests
from typing import List, Tuple, Dict, Any
from config import TORII_API_KEY, TORII_API_URL, TORII_API_HOST, TORII_API_TYPE

class OCRResult:
    def __init__(self, boxes):
        # boxes: List[Dict] where each has {"text": str, "bbox": [x,y,w,h]}
        self.boxes = boxes

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ocr_image(path: str) -> OCRResult:
    if not TORII_API_KEY or not TORII_API_URL:
        raise RuntimeError("TORII_API_KEY/TORII_API_URL not set in .env")

    # ⚠️ Хэрвээ 400/415 авбал доорх түлхүүрийг "image" болгож туршиж болно
    payload = {"image_base64": image_to_base64(path)}

    if TORII_API_TYPE == "rapidapi":
        if not TORII_API_HOST:
            raise RuntimeError("TORII_API_HOST required for RapidAPI")
        headers = {
            "X-RapidAPI-Key": TORII_API_KEY,
            "X-RapidAPI-Host": TORII_API_HOST,
            "Content-Type": "application/json"
        }
    else:
        headers = {
            "Authorization": f"Bearer {TORII_API_KEY}",
            "Content-Type": "application/json"
        }

    r = requests.post(TORII_API_URL, json=payload, headers=headers, timeout=60)
    if r.status_code == 401:
        raise RuntimeError(f"Unauthorized (401). Check TORII_API_TYPE={TORII_API_TYPE}, host/key.")
    r.raise_for_status()
    data = r.json()

    raw_boxes = data.get("boxes", [])
    # normalize
    boxes = []
    for b in raw_boxes:
        text = (b.get("text") or "").strip()
        bbox = b.get("bbox") or [0,0,0,0]
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        boxes.append({"text": text, "bbox": bbox})

    return OCRResult(boxes)
