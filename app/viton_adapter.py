import os
import subprocess
import sys
import tempfile
from typing import Optional
from PIL import Image
import io
import json
import numpy as np


class VITONAdapter:
    def __init__(self, viton_root: str = "/opt/VITON-HD", checkpoints_dir: str = "/opt/VITON-HD/checkpoints"):
        self.viton_root = viton_root
        self.checkpoints_dir = checkpoints_dir

    def run(self, person_img: Image.Image, cloth_img: Image.Image) -> Optional[Image.Image]:
        """Run VITON-HD inference inline by importing their modules, otherwise fallback to subprocess or a simple overlay.

        Returns the output PIL Image or None on failure.
        """
        # Check for installed repo files
        test_script = os.path.join(self.viton_root, "test.py")
        if not os.path.exists(test_script):
            # fallback: simple overlay
            return self._overlay(person_img, cloth_img)

        # Prefer inline import/invocation instead of subprocess. This requires VITON modules to be on PYTHONPATH.
        sys.path.insert(0, self.viton_root)
        inline_success = False
        try:
            from networks import SegGenerator, GMM, ALIASGenerator  # type: ignore
            from datasets import VITONDataset, VITONDataLoader  # type: ignore
            from utils import load_checkpoint, gen_noise  # type: ignore
            inline_success = True
        except Exception as e:
            inline_success = False
            # Fall back to subprocess if inline import fails
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "datasets")
            os.makedirs(data_dir, exist_ok=True)
            # Save images and derive a cloth mask + parse + openpose JSON to satisfy VITON dataset
            person_path = os.path.join(data_dir, "person.jpg")
            cloth_path = os.path.join(data_dir, "cloth.jpg")
            person_img.save(person_path)
            cloth_img.save(cloth_path)
            # Create a simple dataset structure following VITON expectations
            test_mode_dir = os.path.join(data_dir, "test")
            os.makedirs(test_mode_dir, exist_ok=True)
            # image, cloth, cloth-mask, openpose-json, openpose-img, image-parse
            for name in ["image", "cloth", "cloth-mask", "openpose-json", "openpose-img", "image-parse"]:
                os.makedirs(os.path.join(test_mode_dir, name), exist_ok=True)

            # save files inside test dir
            person_test_path = os.path.join(test_mode_dir, "image", "person.jpg")
            cloth_test_path = os.path.join(test_mode_dir, "cloth", "cloth.jpg")
            person_img.save(person_test_path)
            cloth_img.save(cloth_test_path)

            # create a simple cloth mask (binary white rectangle roughly matching cloth size)
            mask = Image.new("L", cloth_img.size, color=255)
            mask_path = os.path.join(test_mode_dir, "cloth-mask", "cloth.jpg")
            mask.save(mask_path)

            # create dummy parse image: simple segmentation: label upper (3), lower (9), head (4)
            parse = Image.new("L", person_img.size, color=3)
            parse_path = os.path.join(test_mode_dir, "image-parse", "person.png")
            parse.save(parse_path)

            # create a fake openpose JSON with 18 keypoints (x,y,confidence), arbitrarily spaced
            width, height = person_img.size
            keypoints = []
            for i in range(18):
                x = int(width * (0.5 + 0.4 * ((i % 6) - 2) / 6.0))
                y = int(height * (0.5 + 0.3 * ((i // 6) - 1) / 3.0))
                keypoints.extend([x, y, 1.0])
            pose_json = {"people": [{"pose_keypoints_2d": keypoints}]}
            pose_json_path = os.path.join(test_mode_dir, "openpose-json", "person_keypoints.json")
            with open(pose_json_path, "w") as fh:
                json.dump(pose_json, fh)

            # create a placeholder openpose-rendered image
            openpose_img = Image.new("RGB", person_img.size, color=(128, 128, 128))
            openpose_img.save(os.path.join(test_mode_dir, "openpose-img", "person_rendered.png"))

            # Create a simple pairs file
            pairs_file = os.path.join(data_dir, "test_pairs.txt")
            with open(pairs_file, "w") as f:
                f.write(f"person.jpg cloth.jpg\n")
            # Prepare an expected dataset structure that test.py might expect. The exact arguments depend on the VITON-HD usage
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir, exist_ok=True)

            if inline_success:
                import torch
                from torch.nn import functional as F
                from torchvision import transforms
                # Prepare opt namespace following test.get_opt defaults
                import argparse
                opt = argparse.Namespace()
                opt.name = "web_infer"
                opt.batch_size = 1
                opt.workers = 0
                opt.load_height = 1024
                opt.load_width = 768
                opt.dataset_dir = data_dir
                opt.dataset_mode = 'test'
                opt.dataset_list = 'test_pairs.txt'
                opt.checkpoint_dir = self.checkpoints_dir
                opt.save_dir = results_dir
                opt.seg_checkpoint = 'seg_final.pth'
                opt.gmm_checkpoint = 'gmm_final.pth'
                opt.alias_checkpoint = 'alias_final.pth'
                opt.semantic_nc = 13

                # Instantiate dataset and loader
                dataset = VITONDataset(opt)
                dataloader = VITONDataLoader(opt, dataset)

                # Instantiate models
                seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
                gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
                alias = ALIASGenerator(opt, input_nc=9)

                def _load_maybe(model, ckpt_name):
                    ck = os.path.join(opt.checkpoint_dir, ckpt_name)
                    if os.path.exists(ck):
                        try:
                            load_checkpoint(model, ck)
                            return True
                        except Exception:
                            return False
                    return False

                ok1 = _load_maybe(seg, opt.seg_checkpoint)
                ok2 = _load_maybe(gmm, opt.gmm_checkpoint)
                ok3 = _load_maybe(alias, opt.alias_checkpoint)

                if not (ok1 and ok2 and ok3):
                    # Missing checkpoints or failed to load
                    return self._overlay(person_img, cloth_img)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                seg.to(device).eval()
                gmm.to(device).eval()
                alias.to(device).eval()

                up = torch.nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
                # soft blur gaussian
                import torchgeometry as tgm
                gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
                gauss.to(device)

                # Get one batch
                try:
                    batch = next(iter(dataloader.data_loader))
                except Exception:
                    return self._overlay(person_img, cloth_img)

                # Move tensors to device
                img_agnostic = batch['img_agnostic'].to(device)
                pose = batch['pose'].to(device)
                c = batch['cloth']['unpaired'].to(device)
                cm = batch['cloth_mask']['unpaired'].to(device)
                parse_agnostic = batch['parse_agnostic'].to(device)

                # Part 1. Segmentation
                parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
                pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
                cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
                seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)
                parse_pred_down = seg(seg_input)
                parse_pred = gauss(up(parse_pred_down))
                parse_pred = parse_pred.argmax(dim=1)[:, None]
                # create parse_old similar to test.py
                parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
                parse_old.scatter_(1, parse_pred.to(device), 1.0)
                labels = {
                    0: ['background', [0, 10]],
                    1: ['hair', [1, 2]],
                    2: ['face', [4, 13]],
                    3: ['upper', [5, 6, 7]],
                    4: ['bottom', [9, 12]],
                    5: ['left_arm', [14]],
                    6: ['right_arm', [15]],
                    7: ['left_leg', [16]],
                    8: ['right_leg', [17]],
                    9: ['left_shoe', [18]],
                    10: ['right_shoe', [19]],
                    11: ['socks', [8]],
                    12: ['noise', [3, 11]]
                }
                parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
                for j in range(len(labels)):
                    for label in labels[j][1]:
                        parse[:, j] += parse_old[:, label]

                # Part 2. Clothes deformation
                agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
                parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
                pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
                c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
                gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
                _, warped_grid = gmm(gmm_input, c_gmm)
                warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
                warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

                # Part 3. Try-on synthesis
                misalign_mask = parse[:, 2:3] - warped_cm
                misalign_mask[misalign_mask < 0.0] = 0.0
                parse_div = torch.cat((parse, misalign_mask), dim=1)
                parse_div[:, 2:3] -= misalign_mask
                output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

                # convert output tensor to PIL image and return
                tensor = (output.clone() + 1) * 0.5 * 255
                tensor = tensor.cpu().clamp(0, 255)
                array = tensor[0].permute(1, 2, 0).cpu().numpy().astype('uint8')
                from PIL import Image as PILImage
                pil = PILImage.fromarray(array)
                return pil
            else:
                # fallback to current subprocess flow for compatibility
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
                try:
                    subprocess.check_call(cmd, cwd=self.viton_root)
                except Exception:
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
