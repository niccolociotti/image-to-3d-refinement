import os
import sys
import threading
import time
import uuid
import traceback
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _candidate_trellis2_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.getenv("TRELLIS2_PATH", ""),
        "/workspace/TRELLIS.2",
        "/workspace/TRELLIS2",
        str(Path.home() / "TRELLIS.2"),
        str(Path.home() / "TRELLIS2"),
        str(project_root.parent / "TRELLIS.2"),
        str(project_root / "TRELLIS.2"),
    ]
    return [Path(path).resolve() for path in candidates if path]


class Model3DGenerationService:
    """
    Generazione modelli 3D da immagine con TRELLIS.2.

    Richiede un server Linux con CUDA e TRELLIS.2 installato nel Python path.
    Setup server consigliato:
        git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
        cd TRELLIS.2
        . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

    Variabili d'ambiente utili:
        TRELLIS2_MODEL_ID=microsoft/TRELLIS.2-4B
        TRELLIS2_DEVICE=cuda
        TRELLIS2_PIPELINE_TYPE=512
        TRELLIS2_MAX_NUM_TOKENS=32768
        TRELLIS2_DECIMATION_TARGET=250000
        TRELLIS2_TEXTURE_SIZE=1024
        TRELLIS2_SIMPLIFY_LIMIT=4000000
        TRELLIS2_REMESH=0
        TRELLIS2_EXTENSION_WEBP=0
    """

    def __init__(self):
        self.pipeline = None
        self._torch = None
        self._o_voxel = None
        self._load_error = None
        self._lock = threading.Lock()
        self.model_loaded = False
        self.model_id = os.getenv("TRELLIS2_MODEL_ID", "microsoft/TRELLIS.2-4B")
        self.device = os.getenv("TRELLIS2_DEVICE", "cuda")
        self.pipeline_type = os.getenv("TRELLIS2_PIPELINE_TYPE", "1024")
        self.max_num_tokens = _env_int("TRELLIS2_MAX_NUM_TOKENS", 65536)
        self.decimation_target = _env_int("TRELLIS2_DECIMATION_TARGET", 750000)
        self.texture_size = _env_int("TRELLIS2_TEXTURE_SIZE", 2048)
        self.simplify_limit = _env_int(
            "TRELLIS2_SIMPLIFY_LIMIT",
            _env_int("TRELLIS2_RESHAPE_SIMPLIFY_LIMIT", 8000000),
        )
        self.remesh = _env_bool("TRELLIS2_REMESH", 1)
        self.extension_webp = _env_bool("TRELLIS2_EXTENSION_WEBP", False)
        self.trellis2_path = self._find_trellis2_path()

    def _find_trellis2_path(self) -> Path | None:
        for path in _candidate_trellis2_paths():
            if (path / "trellis2").is_dir():
                return path
        return None

    def _add_trellis2_to_pythonpath(self) -> None:
        if not self.trellis2_path:
            return

        trellis2_path = str(self.trellis2_path)
        if trellis2_path not in sys.path:
            sys.path.insert(0, trellis2_path)
        os.environ.setdefault("PYTHONPATH", trellis2_path)

    def _patch_trellis_rembg_model(self) -> None:
        """
        TRELLIS.2 ships a gated default rembg model (briaai/RMBG-2.0).
        Override it with a public model so startup works from a local cache.
        """
        try:
            from trellis2.pipelines import rembg as trellis_rembg
        except Exception:
            return

        original_init = trellis_rembg.BiRefNet.__init__
        fallback_model_name = os.getenv("TRELLIS2_REMBG_MODEL_NAME", "ZhengPeng7/BiRefNet")

        if getattr(original_init, "_trellis2_patched", False):
            return

        def patched_init(self, model_name: str = "ZhengPeng7/BiRefNet", *args, **kwargs):
            if model_name == "briaai/RMBG-2.0":
                model_name = fallback_model_name
            return original_init(self, model_name=model_name, *args, **kwargs)

        def patched_call(self, image):
            import torch

            try:
                model_param = next(self.model.parameters())
                target_device = model_param.device
                target_dtype = model_param.dtype
            except Exception:
                target_device = "cuda"
                target_dtype = None

            image_size = image.size
            input_images = self.transform_image(image).unsqueeze(0).to(target_device)
            if target_dtype is not None:
                input_images = input_images.to(dtype=target_dtype)

            with torch.no_grad():
                preds = self.model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = trellis_rembg.transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)
            image.putalpha(mask)
            return image

        patched_init._trellis2_patched = True  # type: ignore[attr-defined]
        trellis_rembg.BiRefNet.__init__ = patched_init
        trellis_rembg.BiRefNet.__call__ = patched_call

    def _patch_trellis_dinov3_model(self) -> None:
        """
        TRELLIS.2 expects a DINOv3 model layout that differs from the current
        transformers implementation. Patch the feature extractor to follow the
        encoder layout exposed by `DINOv3ViTModel`.
        """
        try:
            from trellis2.modules import image_feature_extractor as trellis_ife
            import torch
            import torch.nn.functional as F
        except Exception:
            return

        original_extract_features = trellis_ife.DinoV3FeatureExtractor.extract_features

        if getattr(original_extract_features, "_trellis2_patched", False):
            return

        def patched_extract_features(self, image: torch.Tensor) -> torch.Tensor:
            image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
            hidden_states = self.model.embeddings(image, bool_masked_pos=None)
            position_embeddings = self.model.rope_embeddings(image)

            encoder_layers = getattr(self.model, "layer", None)
            if encoder_layers is None:
                encoder_layers = getattr(getattr(self.model, "model", None), "layer", None)
            if encoder_layers is None:
                raise AttributeError("DINOv3 model does not expose layer blocks")

            for layer_module in encoder_layers:
                hidden_states = layer_module(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )

            return F.layer_norm(hidden_states, hidden_states.shape[-1:])

        patched_extract_features._trellis2_patched = True  # type: ignore[attr-defined]
        trellis_ife.DinoV3FeatureExtractor.extract_features = patched_extract_features

    def load_model(self):
        if self.model_loaded:
            return

        if self._load_error is not None:
            raise RuntimeError(f"Modello 3D non disponibile: {self._load_error}") from self._load_error

        with self._lock:
            if self.model_loaded:
                return

            if self._load_error is not None:
                raise RuntimeError(f"Modello 3D non disponibile: {self._load_error}") from self._load_error

            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            self._add_trellis2_to_pythonpath()

            project_root = Path(__file__).resolve().parents[1]
            default_cache = os.getenv(
                "TRELLIS2_CACHE_DIR", str(project_root / "models" / "trellis_cache")
            )
            if default_cache and Path(default_cache).exists():
                os.environ.setdefault("HF_HOME", default_cache)
                os.environ.setdefault("TRANSFORMERS_CACHE", default_cache)
                os.environ.setdefault("XDG_CACHE_HOME", default_cache)
                os.environ.setdefault("TRELLIS2_LOCAL_SNAPSHOT", default_cache)
                os.environ.setdefault("TRELLIS2_CACHE_DIR", default_cache)

            # Propagate HF_TOKEN into HUGGINGFACE_HUB_TOKEN if present
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
                os.environ.setdefault("HF_TOKEN", hf_token)

            try:
                import torch

                if self.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA non disponibile: TRELLIS.2 richiede una GPU NVIDIA sul server.")

                import o_voxel
                from trellis2.pipelines import Trellis2ImageTo3DPipeline

                self._patch_trellis_rembg_model()
                self._patch_trellis_dinov3_model()

                print(
                    "[Model3DGenerationService] Caricamento TRELLIS.2 "
                    f"model_id={self.model_id} device={self.device}",
                    flush=True,
                )

                # If the user has downloaded a local snapshot, prefer that and avoid network
                local_snapshot = os.getenv(
                    "TRELLIS2_LOCAL_SNAPSHOT",
                    os.getenv("TRELLIS2_CACHE_DIR", default_cache),
                )
                pipeline = None
                if local_snapshot:
                    p = Path(local_snapshot)
                    # If the path points to a directory with pipeline.json or model files, try loading from it
                    if p.exists() and p.is_dir():
                        # heuristic: pipeline.json is present in TRELLIS snapshots
                        if (p / "pipeline.json").exists() or any(p.glob("*.safetensor")):
                            try:
                                print(f"[Model3DGenerationService] Loading TRELLIS.2 from local snapshot: {p}", flush=True)
                                pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(p), local_files_only=True)
                            except Exception:
                                print(f"[Model3DGenerationService] Failed loading local snapshot {p}, falling back to hub", flush=True)
                                traceback.print_exc()

                if pipeline is None:
                    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model_id)
                if self.device == "cuda":
                    print(
                        "[Model3DGenerationService] CUDA pronta: "
                        f"available={torch.cuda.is_available()} "
                        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}",
                        flush=True,
                    )
                    pipeline.cuda()
                    print(
                        "[Model3DGenerationService] Pipeline spostata su CUDA: "
                        f"allocated={torch.cuda.memory_allocated()}",
                        flush=True,
                    )
                elif hasattr(pipeline, "to"):
                    pipeline.to(self.device)

                self.pipeline = pipeline
                self._torch = torch
                self._o_voxel = o_voxel
                self._load_error = None
                self.model_loaded = True
                print("[Model3DGenerationService] TRELLIS.2 caricato.", flush=True)
            except Exception as exc:
                self._load_error = exc
                raise RuntimeError(f"Impossibile caricare TRELLIS.2: {exc}") from exc

    def _load_image(self, image_path: str) -> Any:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")

        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError("Pillow non e' installato: esegui `pip install Pillow`.") from exc

        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        return image

    def _extract_mesh(self, output: Any) -> Any:
        if isinstance(output, (list, tuple)) and output:
            return output[0]
        if isinstance(output, dict) and "mesh" in output:
            return output["mesh"]
        return output

    def _is_cuda_oom(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or isinstance(exc, self._torch.OutOfMemoryError)

    def _export_glb(self, mesh: Any, model3d_path: str) -> None:
        if self.simplify_limit and hasattr(mesh, "simplify"):
            mesh.simplify(self.simplify_limit)

        glb = self._o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=self.decimation_target,
            texture_size=self.texture_size,
            remesh=self.remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )

        try:
            glb.export(model3d_path, extension_webp=self.extension_webp)
        except TypeError:
            glb.export(model3d_path)

    def generate_from_image(
        self,
        image_path: str,
        output_dir: str,
        prompt: str = "",
    ) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model3d_filename = f"model3d_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.glb"
        model3d_path = os.path.join(output_dir, model3d_filename)

        if self._load_error is not None and not self.model_loaded:
            raise RuntimeError(f"Modello 3D non disponibile: {self._load_error}") from self._load_error

        if not self.model_loaded:
            self.load_model()

        image = self._load_image(image_path)
        if prompt:
            print(
                "[Model3DGenerationService] Prompt ricevuto ma TRELLIS.2 image-to-3D usa "
                "l'immagine come condizionamento principale.",
                flush=True,
            )

        print(
            "[Model3DGenerationService] Generazione 3D start "
            f"image={image_path!r} output={model3d_path!r}",
            flush=True,
        )

        def run_pipeline(pipeline_type: str, max_num_tokens: int):
            with self._torch.inference_mode():
                if self.device == "cuda":
                    self._torch.cuda.empty_cache()
                    self._torch.cuda.ipc_collect()
                return self._extract_mesh(
                    self.pipeline.run(
                        image,
                        pipeline_type=pipeline_type,
                        max_num_tokens=max_num_tokens,
                    )
                )

        try:
            mesh = run_pipeline(self.pipeline_type, self.max_num_tokens)
        except Exception as exc:
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
            if self._is_cuda_oom(exc):
                raise RuntimeError(
                    "CUDA out of memory durante la generazione 3D con il profilo leggero. "
                    "Libera VRAM o riduci il carico della GPU e riprova."
                ) from exc
            raise

        self._export_glb(mesh, model3d_path)

        print(f"[Model3DGenerationService] GLB salvato: {model3d_path}", flush=True)
        return model3d_path
