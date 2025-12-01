import torch
from torchvision import transforms
from PIL import Image
from viton_adapter import VITONAdapter


class VITONModel:
    """Adapter-based VITON model class used by the app.
    It will use VITON-HD (if installed) or fall back to a simple overlay for demo.
    """
    def __init__(self, device="cuda"):
        print("Loading VITON-HD adapter...")
        self.device = device
        self.adapter = VITONAdapter()

        self.preprocess = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor()
        ])

    def run(self, person_img: Image.Image, cloth_img: Image.Image):
        # Use the adapter to produce an output PIL Image
        output = self.adapter.run(person_img, cloth_img)
        if output is None:
            raise RuntimeError("VITON inference failed or no output was created")
        return output
