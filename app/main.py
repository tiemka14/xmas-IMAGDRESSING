from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.model import IMAGDressingModel
from PIL import Image
import io
import base64

app = FastAPI(title="IMAGDressing API")

# Allow your DigitalOcean frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your DO domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = IMAGDressingModel(device="cuda")

@app.post("/tryon")
async def tryon(person: UploadFile = File(...), cloth: UploadFile = File(...)):
    person_img = Image.open(io.BytesIO(await person.read())).convert("RGB")
    cloth_img = Image.open(io.BytesIO(await cloth.read())).convert("RGB")

    output = model.run(person_img, cloth_img)

    # Convert to base64 to send JSON back
    buffered = io.BytesIO()
    output.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"result": img_str}
