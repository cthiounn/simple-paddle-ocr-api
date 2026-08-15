from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import io

app = FastAPI(
    title="POI OCR API",
    version="1.0.1",
)

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ocr")
def ocr_image(file: bytes = File(...)):
    if not file:
        raise HTTPException(
            status_code=400,
            detail="Empty file",
        )

    try:
        image = Image.open(
            io.BytesIO(file)
        ).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    image_np = np.asarray(image)

    try:
        results = ocr.predict(image_np)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PaddleOCR error: {exc}",
        )

    texts = []

    for result in results:
        rec_texts = result["rec_texts"]
        rec_scores = result["rec_scores"]
        rec_boxes = result["rec_boxes"]

        for text, score, box in zip(
            rec_texts,
            rec_scores,
            rec_boxes,
        ):
            if not text:
                continue

            x1, y1, x2, y2 = map(int, box)

            texts.append({
                "text": text,
                "confidence": float(score),
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "center": [
                    (x1 + x2) // 2,
                    (y1 + y2) // 2,
                ],
            })

    return {
        "width": image.width,
        "height": image.height,
        "count": len(texts),
        "texts": texts,
    }
