import os
import subprocess
import tempfile
from typing import Optional
from PIL import Image


class VITONAdapter:
    def __init__(self, viton_root: str = "/opt/VITON-HD", checkpoints_dir: str = "/opt/VITON-HD/checkpoints"):
        self.viton_root = viton_root
        self.checkpoints_dir = checkpoints_dir

    def run(self, person_img: Image.Image, cloth_img: Image.Image) -> Optional[Image.Image]:
        """Run VITON-HD inference as a subprocess test.py invocation. Falls back to a simple overlay if not configured.

        Returns the output PIL Image or None on failure.
        """
        # Check for certificate of installed repo
        test_script = os.path.join(self.viton_root, "test.py")
        if not os.path.exists(test_script):
            # fallback: simple overlay
            return self._overlay(person_img, cloth_img)

        # Try to run a single pair job using test.py. We need to prepare a test dataset.
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "datasets")
            os.makedirs(data_dir, exist_ok=True)
            # Save images
            person_path = os.path.join(data_dir, "person.jpg")
            cloth_path = os.path.join(data_dir, "cloth.jpg")
            person_img.save(person_path)
            cloth_img.save(cloth_path)
            # Create a simple test_pairs.txt
            pairs_file = os.path.join(data_dir, "test_pairs.txt")
            with open(pairs_file, "w") as f:
                f.write(f"person.jpg cloth.jpg\n")
            # Prepare an expected dataset structure that test.py might expect. The exact arguments depend on the VITON-HD usage
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir, exist_ok=True)

            cmd = [
                "python",
                test_script,
                "--name",
                "web_infer",
                "--datapath",
                data_dir,
                "--results_dir",
                results_dir,
                "--save_dir",
                results_dir,
            ]
            # When VITON-HD is installed, run via subprocess
            try:
                subprocess.check_call(cmd, cwd=self.viton_root)
            except Exception:
                # fallback on error
                return self._overlay(person_img, cloth_img)

            # Attempt to read the first result
            # Search results folder for an image
            for root, dirs, files in os.walk(results_dir):
                for fname in files:
                    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                        try:
                            return Image.open(os.path.join(root, fname)).convert('RGB')
                        except Exception:
                            continue
            return None

    def _overlay(self, person_img: Image.Image, cloth_img: Image.Image) -> Image.Image:
        # A very naive overlay: center the cloth over the person with alpha blend.
        person = person_img.convert('RGBA')
        cloth = cloth_img.convert('RGBA')
        cloth = cloth.resize((person.width // 2, person.height // 2))
        pos = ((person.width - cloth.width) // 2, (person.height - cloth.height) // 2)
        result = Image.new('RGBA', person.size)
        result.paste(person, (0, 0))
        result.paste(cloth, pos, cloth)
        return result.convert('RGB')
