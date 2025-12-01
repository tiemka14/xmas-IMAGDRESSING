import torch
from torchvision import transforms
from PIL import Image
from imagdressing.pipelines.IMAGDressing_v1_pipeline import IMAGDressing_v1 as IMAGDressingPipeline


class IMAGDressingModel:
    def __init__(self, device="cuda"):
        print("Loading IMAGDressing model...")
        self.device = device

        # Load model (modify path if needed)
        self.pipe = IMAGDressingPipeline.from_pretrained(
            "IMAGDressing/IMAGDressing",
            torch_dtype=torch.float16,
        ).to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor()
        ])

    def run(self, person_img: Image.Image, cloth_img: Image.Image):
        person_tensor = self.preprocess(person_img).unsqueeze(0).to(self.device)
        cloth_tensor = self.preprocess(cloth_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            result = self.pipe(
                person_image=person_tensor,
                garment_image=cloth_tensor,
            )

        output = result.images[0]
        return output
