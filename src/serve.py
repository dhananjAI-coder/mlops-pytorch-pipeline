import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model


app = FastAPI(
    title="CIFAR-10 Model Serving API",
    version="1.0.0",
)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/checkpoints/classifier_v1.pt",
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


model = None


preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_model():
    global model

    model = get_model(
        architecture="cnn",
        num_classes=10,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except Exception as exc:
        print(
            f"Failed to load model: {exc}",
            flush=True,
        )


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    try:
        image_bytes = await image.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        tensor = preprocess(image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

        results = [
            {
                "class": CLASS_NAMES[i],
                "probability": round(
                    probabilities[i].item(),
                    6,
                ),
            }
            for i in range(len(CLASS_NAMES))
        ]

        results.sort(
            key=lambda x: x["probability"],
            reverse=True,
        )

        return {
            "prediction": results[0]["class"],
            "probability": results[0]["probability"],
            "class_probabilities": results,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )