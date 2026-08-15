from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import io
from paddleocr import TextDetection, TextRecognition


app = FastAPI(
    title="POI OCR API",
    version="1.0.1",
)

@app.get("/")
def health():
    return {"status": "ok"}


# ============================================================
# Models
# ============================================================

detector = TextDetection(
    thresh=0.2,
    box_thresh=0.3,
    limit_side_len=1600,
    limit_type="max",
)
recognizer = TextRecognition()


# ============================================================
# OCR endpoint
# ============================================================

@app.post("/ocr")
def ocr_image(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Content-Type missing",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:
        data = file.file.read()

        image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    image_np = np.asarray(image)

    height, width = image_np.shape[:2]

    # --------------------------------------------------------
    # Text detection
    # --------------------------------------------------------

    try:
        detection_results = detector.predict(image_np)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Text detection error: {exc}",
        )

    texts = []

    # --------------------------------------------------------
    # Detection → crop → recognition
    # --------------------------------------------------------

    for detection_result in detection_results:

        data = detection_result.json

        if isinstance(data, str):
            import json
            data = json.loads(data)

        res = data["res"]

        polygons = res.get("dt_polys", [])

        for polygon in polygons:

            polygon = np.asarray(polygon)

            # ----------------------------------------------
            # Polygon → bounding box
            # ----------------------------------------------

            x1 = max(
                0,
                int(polygon[:, 0].min()),
            )

            y1 = max(
                0,
                int(polygon[:, 1].min()),
            )

            x2 = min(
                width,
                int(polygon[:, 0].max()),
            )

            y2 = min(
                height,
                int(polygon[:, 1].max()),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # ----------------------------------------------
            # Crop
            # ----------------------------------------------

            crop = image_np[
                y1:y2,
                x1:x2,
            ]

            if crop.size == 0:
                continue

            # ----------------------------------------------
            # Recognition
            # ----------------------------------------------

            try:
                recognition_results = recognizer.predict(
                    crop
                )

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Text recognition error: {exc}",
                )

            for recognition_result in recognition_results:

                rec_data = recognition_result.json

                if isinstance(rec_data, str):
                    import json
                    rec_data = json.loads(rec_data)

                rec = rec_data["res"]

                text = rec.get(
                    "rec_text",
                    "",
                )

                score = rec.get(
                    "rec_score",
                    0.0,
                )

                if not text:
                    continue

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

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "width": width,
        "height": height,
        "count": len(texts),
        "texts": texts,
    }