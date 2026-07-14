import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from services.comfyui_bootstrap import (
    add_comfyui_to_path,
    comfyui_not_found_message,
    patch_torch_for_comfyui,
)

add_comfyui_to_path()
patch_torch_for_comfyui(torch)

# ComfyUI nodes — disponibili solo se ComfyUI è nel PYTHONPATH o in COMFYUI_PATH
try:
    from nodes import NODE_CLASS_MAPPINGS

    UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
    VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
    CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    VAEEncodeForInpaint = NODE_CLASS_MAPPINGS["VAEEncodeForInpaint"]()
    KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
    VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

    COMFYUI_AVAILABLE = True
    COMFYUI_IMPORT_ERROR = None
except Exception as exc:
    COMFYUI_AVAILABLE = False
    COMFYUI_IMPORT_ERROR = exc


class InpaintService:
    def __init__(self):
        self.unet = None
        self.clip = None
        self.vae = None
        self.models_loaded = False

    def load_models(self, checkpoint=None):
        if not COMFYUI_AVAILABLE:
            detail = f" Dettaglio: {COMFYUI_IMPORT_ERROR}" if COMFYUI_IMPORT_ERROR else ""
            raise RuntimeError(f"{comfyui_not_found_message()}{detail}")

        checkpoint = checkpoint or os.getenv(
            "Z_IMAGE_CHECKPOINT",
            "z-image-turbo-fp8-e4m3fn.safetensors",
        )
        clip_name = os.getenv("Z_IMAGE_CLIP", "qwen_3_4b.safetensors")
        vae_name = os.getenv("Z_IMAGE_VAE", "ae.safetensors")

        self.unet = UNETLoader.load_unet(checkpoint, "fp8_e4m3fn_fast")[0]
        self.clip = CLIPLoader.load_clip(clip_name, type="lumina2")[0]
        self.vae = VAELoader.load_vae(vae_name)[0]

        self.models_loaded = True

        print(f"[InpaintService] Modelli caricati: {checkpoint}, {clip_name}, {vae_name}")

    def _ensure_loaded(self):
        if not self.models_loaded:
            self.load_models()

    def _load_image_tensor(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image).astype(np.float32) / 255.0

        # ComfyUI image format: [batch, height, width, channels]
        image_tensor = torch.from_numpy(image_np)[None, ...]

        return image_tensor, image.size

    def _make_odd_kernel_size(self, radius: int) -> int:
        """
        ImageFilter.MaxFilter richiede un kernel dispari.
        grow_mask_by=8 -> kernel 17.
        """
        radius = max(0, int(radius))
        return radius * 2 + 1

    def _load_masks(
        self,
        mask_path: str,
        image_size,
        invert_mask: bool = False,
        grow_mask_by: int = 8,
        mask_blur: float = 4.0,
        mask_threshold: int = 128,
    ):
        """
        Ritorna due maschere:

        1. model_mask_tensor:
           - mask dura/binaria
           - usata da VAEEncodeForInpaint
           - più precisa per il modello

        2. blend_mask:
           - mask allargata e sfumata
           - usata solo per fondere immagine originale e output finale
           - evita stacchi netti sui bordi

        Convenzione:
        - bianco = area da modificare
        - nero = area da preservare
        """
        mask = Image.open(mask_path).convert("L")

        # Per le mask è meglio NEAREST, così non crea grigi strani in resize.
        mask = mask.resize(image_size, Image.NEAREST)

        if invert_mask:
            mask = ImageOps.invert(mask)

        # Binarizza la mask.
        # Sopra threshold diventa bianco, sotto threshold diventa nero.
        hard_mask = mask.point(
            lambda p: 255 if p >= mask_threshold else 0
        ).convert("L")

        # Mask per il modello: precisa e non sfumata.
        model_mask_np = np.asarray(hard_mask).astype(np.float32) / 255.0

        # ComfyUI mask format: [batch, height, width]
        model_mask_tensor = torch.from_numpy(model_mask_np)[None, ...]

        # Mask per blending finale: possiamo allargarla e sfumarla.
        blend_mask = hard_mask.copy()

        if grow_mask_by and grow_mask_by > 0:
            kernel_size = self._make_odd_kernel_size(grow_mask_by)
            blend_mask = blend_mask.filter(ImageFilter.MaxFilter(kernel_size))

        if mask_blur and mask_blur > 0:
            blend_mask = blend_mask.filter(
                ImageFilter.GaussianBlur(radius=mask_blur)
            )

        return model_mask_tensor, blend_mask

    def _decoded_to_pil(self, decoded: torch.Tensor) -> Image.Image:
        decoded = decoded.detach().clamp(0, 1).cpu().numpy()
        image_np = np.array(decoded * 255, dtype=np.uint8)[0]
        return Image.fromarray(image_np)

    def _composite_inpaint_result(
        self,
        original_image_path: str,
        generated_image: Image.Image,
        blend_mask: Image.Image,
    ) -> Image.Image:
        """
        Fonde l'immagine originale con quella generata.

        Dove blend_mask è bianca:
        - prende generated_image

        Dove blend_mask è nera:
        - tiene original_image

        Dove blend_mask è grigia:
        - fa blending morbido
        """
        original_image = Image.open(original_image_path).convert("RGB")
        original_image = original_image.resize(generated_image.size, Image.LANCZOS)

        blend_mask = blend_mask.resize(generated_image.size, Image.LANCZOS)

        final_image = Image.composite(
            generated_image,
            original_image,
            blend_mask,
        )

        return final_image

    def _normalize_legacy_defaults(self, steps, cfg, denoise, grow_mask_by, mask_blur):
        steps = int(steps)
        cfg = float(cfg)
        denoise = float(denoise)
        grow_mask_by = int(grow_mask_by)
        mask_blur = float(mask_blur)

        if steps < 30:
            steps = 30
        if cfg < 5.0:
            cfg = 7.0
        if denoise < 0.15:
            denoise = 0.15
        if grow_mask_by >= 8:
            grow_mask_by = 0
        if mask_blur > 1.0:
            mask_blur = 0.5

        return steps, cfg, denoise, grow_mask_by, mask_blur

    @torch.inference_mode()
    def inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        negative_prompt: str = "blurry, ugly, bad, artifacts, distorted, deformed, warped, resized, cropped, extra parts",
        steps: int = 20,
        cfg: float = 8.0,
        seed: int = 0,
        denoise: float = 0.8,
        grow_mask_by: int = 2,
        mask_blur: float = 1.0,
        invert_mask: bool = False,
        mask_threshold: int = 128,
        output_dir: str = "/tmp/cg_pipeline/outputs",
    ) -> str:
        self._ensure_loaded()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if seed == 0:
            seed = random.randint(0, 18_446_744_073_709_551_615)

        image, image_size = self._load_image_tensor(image_path)

        mask, blend_mask = self._load_masks(
            mask_path=mask_path,
            image_size=image_size,
            invert_mask=invert_mask,
            grow_mask_by=grow_mask_by,
            mask_blur=mask_blur,
            mask_threshold=mask_threshold,
        )

        positive = CLIPTextEncode.encode(self.clip, prompt)[0]
        negative = CLIPTextEncode.encode(self.clip, negative_prompt)[0]

        latent_image = VAEEncodeForInpaint.encode(
            self.vae,
            image,
            mask,
            grow_mask_by,
        )[0]

        samples = KSampler.sample(
            self.unet,
            seed,
            steps,
            cfg,
            "euler",
            "simple",
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )[0]

        decoded = VAEDecode.decode(self.vae, samples)[0].detach()

        generated_image = self._decoded_to_pil(decoded)

        final_image = self._composite_inpaint_result(
            original_image_path=image_path,
            generated_image=generated_image,
            blend_mask=blend_mask,
        )

        out_path = os.path.join(output_dir, f"inpainted_{seed}.png")
        final_image.save(out_path)

        print(f"[InpaintService] Immagine salvata: {out_path}")

        return out_path