import base64
import os
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_sam2_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.getenv("SAM2_PATH", ""),
        "/workspace/sam2",
        str(Path.home() / "sam2"),
        str(project_root.parent / "sam2"),
        str(project_root / "sam2"),
    ]
    return [Path(path).resolve() for path in candidates if path]


def _candidate_checkpoint_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    sam2_path = os.getenv("SAM2_PATH", "")
    checkpoint_dir = os.getenv("SAM2_CHECKPOINT_DIR", "")
    checkpoint_names = [
        "sam2.1_hiera_base_plus.pt",
        "sam2_hiera_base_plus.pt",
    ]
    candidates = [
        os.getenv("SAM2_CHECKPOINT", ""),
        str(project_root / "models" / "sam2" / "sam2.1_hiera_base_plus.pt"),
        str(project_root / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
        str(project_root.parent / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
    ]

    if checkpoint_dir:
        candidates.extend(str(Path(checkpoint_dir) / name) for name in checkpoint_names)

    if sam2_path:
        candidates.extend(str(Path(sam2_path) / "checkpoints" / name) for name in checkpoint_names)

    return [Path(path).resolve() for path in candidates if path]


class SAM2SegmentationService:


    def __init__(self):
        self.predictor = None
        self._torch = None
        self._load_error = None
        self._lock = threading.Lock()
        self.model_loaded = False
        self.sam2_path = self._find_sam2_path()
        self.model_cfg = os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
        self.checkpoint = self._find_checkpoint_path()
        self.device = os.getenv("SAM2_DEVICE", "auto")

    def _find_sam2_path(self) -> Path | None:
        for path in _candidate_sam2_paths():
            if (path / "sam2").is_dir():
                return path
        return None

    def _find_checkpoint_path(self) -> Path | None:
        for path in _candidate_checkpoint_paths():
            if path.is_file():
                return path
        return None

    def _add_sam2_to_pythonpath(self) -> None:
        if not self.sam2_path:
            return

        sam2_path = str(self.sam2_path)
        if sam2_path not in sys.path:
            sys.path.insert(0, sam2_path)
        os.environ.setdefault("PYTHONPATH", sam2_path)

    def _resolve_device(self, torch_module):
        requested = self.device.strip().lower()
        if requested and requested != "auto":
            return requested

        if torch_module.cuda.is_available():
            return "cuda"

        if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return "mps"

        return "cpu"

    def load_model(self):
        if self.model_loaded:
            return

        if self._load_error is not None:
            raise RuntimeError(f"SAM2 non disponibile: {self._load_error}") from self._load_error

        with self._lock:
            if self.model_loaded:
                return

            if self._load_error is not None:
                raise RuntimeError(f"SAM2 non disponibile: {self._load_error}") from self._load_error

            try:
                self._add_sam2_to_pythonpath()

                import torch
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                if self.checkpoint is None:
                    self.checkpoint = self._find_checkpoint_path()
                if self.checkpoint is None:
                    searched_paths = "\n".join(
                        f"- {path}" for path in _candidate_checkpoint_paths()
                    )
                    raise RuntimeError(
                        "Checkpoint SAM2 non trovato. Imposta SAM2_CHECKPOINT "
                        "oppure salva sam2.1_hiera_base_plus.pt in models/sam2/.\n"
                        f"Path cercati:\n{searched_paths}"
                    )

                device = self._resolve_device(torch)
                print(
                    "[SAM2SegmentationService] Caricamento SAM2 "
                    f"cfg={self.model_cfg} checkpoint={self.checkpoint} device={device}",
                    flush=True,
                )

                model = build_sam2(self.model_cfg, str(self.checkpoint), device=device)
                self.predictor = SAM2ImagePredictor(model)
                self._torch = torch
                self.device = device
                self.model_loaded = True
            except Exception as exc:
                if "Checkpoint SAM2 non trovato" not in str(exc):
                    self._load_error = exc
                raise RuntimeError(f"Impossibile caricare SAM2: {exc}") from exc

    def _ensure_loaded(self):
        if not self.model_loaded:
            self.load_model()

    def _inference_context(self):
        torch = self._torch
        if torch is None:
            return nullcontext()

        inference_mode = torch.inference_mode()
        if self.device == "cuda":
            return torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)

        return inference_mode

    def _load_image(self, image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
        image = Image.open(image_path).convert("RGB")
        return np.asarray(image), image.size

    def _normalize_points(
        self,
        points,
        image_size: tuple[int, int],
        coordinates_normalized: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not points:
            return None, None

        width, height = image_size
        coords = []
        labels = []
        for point in points:
            if isinstance(point, dict):
                x = point.get("x")
                y = point.get("y")
                label = point.get("label", 1)
            else:
                x = point[0]
                y = point[1]
                label = point[2] if len(point) > 2 else 1

            if x is None or y is None:
                raise ValueError("Ogni punto deve contenere x e y.")

            if coordinates_normalized:
                x = float(x) * width
                y = float(y) * height

            coords.append([float(x), float(y)])
            labels.append(int(label))

        return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    def _normalize_box(self, box, image_size: tuple[int, int], coordinates_normalized: bool = False):
        if not box:
            return None

        if isinstance(box, dict):
            values = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
        else:
            values = list(box)

        if len(values) != 4 or any(value is None for value in values):
            raise ValueError("box deve essere [x1, y1, x2, y2].")

        if coordinates_normalized:
            width, height = image_size
            values = [
                float(values[0]) * width,
                float(values[1]) * height,
                float(values[2]) * width,
                float(values[3]) * height,
            ]

        return np.asarray(values, dtype=np.float32)

    def _pick_mask(self, masks, scores, mask_index: int | None, selection_mode: str = "part"):
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM2 non ha restituito maschere.")

        if mask_index is not None:
            index = max(0, min(int(mask_index), len(masks) - 1))
        elif selection_mode == "part":
            areas = [int(np.count_nonzero(mask)) for mask in masks]
            non_empty = [(area, idx) for idx, area in enumerate(areas) if area > 0]
            index = min(non_empty)[1] if non_empty else 0
        elif selection_mode == "largest":
            areas = [int(np.count_nonzero(mask)) for mask in masks]
            index = int(np.argmax(areas))
        elif scores is not None and len(scores) > 0:
            index = int(np.argmax(scores))
        else:
            index = 0

        return masks[index], index

    def segment(
        self,
        image_path: str,
        points=None,
        box=None,
        multimask_output: bool = True,
        mask_index: int | None = None,
        output_dir: str = "/tmp/cg_pipeline/outputs",
        output_name: str = "mask.png",
        invert_mask: bool = False,
        grow_mask_by: int = 0,
        mask_blur: float = 0.0,
        coordinates_normalized: bool = False,
        selection_mode: str = "part",
    ) -> dict:
        self._ensure_loaded()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        image_np, image_size = self._load_image(image_path)
        point_coords, point_labels = self._normalize_points(
            points,
            image_size=image_size,
            coordinates_normalized=coordinates_normalized,
        )
        box_array = self._normalize_box(
            box,
            image_size=image_size,
            coordinates_normalized=coordinates_normalized,
        )

        if point_coords is None and box_array is None:
            raise ValueError("Serve almeno un punto touch oppure una box.")

        context = self._inference_context()
        if isinstance(context, tuple):
            with context[0], context[1]:
                self.predictor.set_image(image_np)
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_array,
                    multimask_output=multimask_output,
                )
        else:
            with context:
                self.predictor.set_image(image_np)
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_array,
                    multimask_output=multimask_output,
                )

        mask, selected_index = self._pick_mask(
            masks,
            scores,
            mask_index,
            selection_mode=selection_mode,
        )
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize(image_size, Image.NEAREST)

        if invert_mask:
            mask_image = Image.fromarray(255 - np.asarray(mask_image), mode="L")

        if grow_mask_by and grow_mask_by > 0:
            kernel_size = int(grow_mask_by) * 2 + 1
            mask_image = mask_image.filter(ImageFilter.MaxFilter(kernel_size))

        if mask_blur and mask_blur > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=float(mask_blur)))

        output_path = str(Path(output_dir) / output_name)
        mask_image.save(output_path)

        score = None
        if scores is not None and len(scores) > selected_index:
            score = float(scores[selected_index])

        mask_base64 = None
        if _env_bool("SAM2_RETURN_BASE64_BY_DEFAULT", False):
            mask_base64 = base64.b64encode(Path(output_path).read_bytes()).decode("utf-8")

        return {
            "mask_path": output_path,
            "mask_base64": mask_base64,
            "score": score,
            "selected_mask_index": selected_index,
            "selection_mode": selection_mode,
            "image_width": image_size[0],
            "image_height": image_size[1],
        }
