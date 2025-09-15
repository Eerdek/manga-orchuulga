import base64, requests
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

class ToriiOCRResponse(BaseModel):
    lines: list  # list of { "text": str, "boundingBox": [...] }

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ocr_image(path: str, api_key: str, api_url: str = "https://torii-ocr.p.rapidapi.com/ocr"):
    if not api_key or not api_url:
        raise RuntimeError("TORII_API_KEY/TORII_API_URL not set")
    with open(path, "rb") as image:
        image_bytes = image.read()

    response = requests.post(
        api_url,
        headers={
            "content-type": "application/octet-stream",
            "x-rapidapi-host": "torii-ocr.p.rapidapi.com",
            "x-rapidapi-key": api_key
        },
        data=image_bytes
    )

    if response.ok:
        data = response.json()
        print("OCR API response:", data)  # <--- Энд хариуг шалгана
        # return ToriiOCRResponse.model_validate(data)
        return data  # Түр зуур dict-ээр буцаана
    else:
        raise Exception(response.text)
