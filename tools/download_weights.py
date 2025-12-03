import os
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download


def prepare_base_model():
    print(f'Preparing base stable-diffusion-v1-5 weights...')
    local_dir = "./pretrained_weights/stable-diffusion-v1-5"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        print(f"[DEBUG prepare_base_model] downloading {hub_file} from runwayml/stable-diffusion-v1-5")
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
            force_download=True
        )


def prepare_image_encoder():
    print(f"Preparing image encoder weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        print(f"[DEBUG prepare_image_encoder] downloading {hub_file} from lambdalabs/sd-image-variations-diffusers")
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
            force_download=True
        )



def prepare_vae():
    print(f"Preparing vae weights...")
    local_dir = "./pretrained_weights/sd-vae-ft-mse"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "config.json",
        "diffusion_pytorch_model.bin",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        print(f"[DEBUG prepare_vae] downloading {hub_file} from stabilityai/sd-vae-ft-mse")
        hf_hub_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
            force_download=True
        )


def prepare_anyone():
    print(f"Preparing AnimateAnyone weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "denoising_unet.pth",
        "motion_module.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        print(f"[DEBUG prepare_anyone] downloading {hub_file} from patrolli/AnimateAnyone")
        hf_hub_download(
            repo_id="patrolli/AnimateAnyone",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
            force_download=True
        )

if __name__ == '__main__':
    prepare_base_model()
    prepare_image_encoder()
    prepare_vae()
    prepare_anyone()
    