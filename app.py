import os
import sys
import signal
import time
import traceback
import base64
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_simple_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _first_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _bootstrap_environment() -> None:
    _load_simple_env_file(PROJECT_ROOT / ".env")

    trellis_cache_dir = PROJECT_ROOT / "models" / "trellis_cache"
    os.environ.setdefault("TRELLIS2_CACHE_DIR", str(trellis_cache_dir))
    os.environ.setdefault("TRELLIS2_LOCAL_SNAPSHOT", str(trellis_cache_dir))

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    os.environ.setdefault("TRELLIS2_PRELOAD", "1")


_bootstrap_environment()

from flask import Flask, request, jsonify, send_file
from utils.storage import RUNS_DIR, ensure_dirs, make_session_dir, save_base64_image
from services.image_service import ImageGenerationService
from services.edit_service2 import InpaintService
#from services.edit_service import InpaintService
#wdsfrom services.edit_service_sdxl_inpaint import InpaintService
from services.sam2_service import SAM2SegmentationService
from services.model3d_service import Model3DGenerationService   


app = Flask(__name__)
ensure_dirs()

image_service  = ImageGenerationService()
edit_service2  = InpaintService()
#edit_service   = InpaintService() 
model3d_service = Model3DGenerationService()
sam2_service = SAM2SegmentationService()

# Carica i modelli una sola volta
try:
    image_service.load_models(os.getenv("Z_IMAGE_CHECKPOINT"))
except Exception as e:
    print(f"[startup] Modelli immagine non caricati: {e}")

#try:
#    qwen_edit_service.load_models()
#except Exception as e:
#    print(f"[startup] Modello Qwen Inpaint non caricato: {e}")

#try:
#    edit_service.load_models(os.getenv("INPAINT_MODEL_ID"))
#except Exception as e:
#    print(f"[startup] Modello inpaint non caricato: {e}")

try:
    model3d_service.load_model()
except Exception as e:
    print(f"[startup] Modello 3D non caricato: {e}")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


def public_file_url(path):
    try:
        relative_path = Path(path).resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    return request.host_url.rstrip("/") + "/files/" + relative_path.as_posix()

def _coerce_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _schedule_auto_shutdown(timeout_seconds: int) -> None:
    if timeout_seconds <= 0:
        return

    def _stop_server() -> None:
        print(f"[app] Auto-shutdown dopo {timeout_seconds}s", flush=True)
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(timeout_seconds, _stop_server)
    timer.daemon = True
    timer.start()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/files/<path:relative_path>")
def get_file(relative_path):
    base_dir = RUNS_DIR.resolve()
    file_path = (base_dir / relative_path).resolve()
    if not file_path.is_file() or base_dir not in file_path.parents:
        return jsonify({"error": "File non trovato."}), 404
    return send_file(file_path)


@app.post("/generate-image")
def generate_image():
    data = request.get_json(force=True)
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))
    started_at = time.time()
    print(
        "[app] /generate-image start "
        f"session={Path(session_dir).name} prompt={data['prompt']!r}",
        flush=True,
    )
    try:
        output_path = image_service.generate(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", "blurry, ugly, bad, background"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 9)),
            cfg=float(data.get("cfg", 1.0)),
            seed=int(data.get("seed", 0)),
            output_dir=session_dir,
        )
    except Exception as e:
        print(f"[app] /generate-image error after {time.time() - started_at:.1f}s: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    print(f"[app] /generate-image done after {time.time() - started_at:.1f}s: {output_path}", flush=True)
    return jsonify({
        "status": "ok",
        "image_path": output_path,
        "image_url": public_file_url(output_path),
    })


@app.post("/edit-image")
def edit_image():
    data = request.get_json(force=True)
    if not data.get("image_path"):
        return jsonify({"error": "Campo 'image_path' obbligatorio."}), 400
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))

    mask_path = data.get("mask_path")
    if data.get("mask_base64"):
        mask_path = save_base64_image(data["mask_base64"], session_dir, "mask.png")
    if not mask_path:
        return jsonify({"error": "Serve 'mask_path' oppure 'mask_base64'."}), 400

    started_at = time.time()
    print(
        "[app] /edit-image start "
        f"session={Path(session_dir).name} image={data['image_path']!r} "
        f"mask={mask_path!r} prompt={data['prompt']!r}",
        flush=True,
    )
    try:
        inpaint_args = dict(
            image_path=data["image_path"],
            mask_path=mask_path,
            prompt=data["prompt"],
            negative_prompt=data.get(
                "negative_prompt",
                "blurry, ugly, bad, distorted, deformed, warped, resized, cropped, extra parts",
            ),
            seed=int(data.get("seed", 0)),
            steps=int(data.get("steps", 20)),
            cfg=float(data.get("cfg", 8.0)),
            denoise=float(data.get("denoise", 0.8)),
            mask_blur=float(data.get("mask_blur", 1.0)),
            mask_threshold=int(data.get("mask_threshold", 128)),
            grow_mask_by=int(data.get("grow_mask_by", data.get("mask_expand", 2))),
            invert_mask=_coerce_bool(data.get("invert_mask"), False),
            output_dir=session_dir,
        )
        output_path = edit_service2.inpaint(**inpaint_args)
    except Exception as e:
        print(f"[app] /edit-image error after {time.time() - started_at:.1f}s: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    print(f"[app] /edit-image done after {time.time() - started_at:.1f}s: {output_path}", flush=True)
    return jsonify({
        "status": "ok",
        "edited_image_path": output_path,
        "edited_image_url": public_file_url(output_path),
    })


@app.post("/generate-3d")
def generate_3d():
    data = request.get_json(force=True)
    if not data.get("image_path"):
        return jsonify({"error": "Campo 'image_path' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))
    try:
        model3d_path = model3d_service.generate_from_image(
            image_path=data["image_path"],
            output_dir=session_dir,
            prompt=data.get("prompt", ""),
        )
    except Exception as e:
        print(f"[app] /generate-3d error: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "ok",
        "model3d_path": model3d_path,
        "model3d_url": public_file_url(model3d_path),
    })

@app.post("/segment-image")
def segment_image():
    data = request.get_json(force=True)

    session_dir = make_session_dir(data.get("session_id"))

    image_path = data.get("image_path")
    if data.get("image_base64"):
        image_path = save_base64_image(data["image_base64"], session_dir, "sam2_input.png")
    if not image_path:
        return jsonify({"error": "Serve 'image_path' oppure 'image_base64'."}), 400

    points = data.get("points")
    if points is None and ("x" in data and "y" in data):
        points = [{"x": data["x"], "y": data["y"], "label": data.get("label", 1)}]

    if not points and not data.get("box"):
        return jsonify({"error": "Serve almeno 'points', oppure x/y, oppure 'box'."}), 400

    started_at = time.time()
    print(
        "[app] /segment-image start "
        f"session={Path(session_dir).name} image={image_path!r} "
        f"points={points!r} box={data.get('box')!r}",
        flush=True,
    )

    try:
        result = sam2_service.segment(
            image_path=image_path,
            points=points,
            box=data.get("box"),
            multimask_output=_coerce_bool(data.get("multimask_output"), True),
            mask_index=data.get("mask_index"),
            output_dir=session_dir,
            output_name=data.get("output_name", "mask.png"),
            invert_mask=_coerce_bool(data.get("invert_mask"), False),
            grow_mask_by=int(data.get("grow_mask_by", 0)),
            mask_blur=float(data.get("mask_blur", 0.0)),
            coordinates_normalized=_coerce_bool(data.get("coordinates_normalized"), False),
            selection_mode=data.get("selection_mode", "part"),
        )
    except Exception as e:
        print(f"[app] /segment-image error after {time.time() - started_at:.1f}s: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    if _coerce_bool(data.get("return_base64"), False) and not result.get("mask_base64"):
        mask_bytes = Path(result["mask_path"]).read_bytes()
        result["mask_base64"] = base64.b64encode(mask_bytes).decode("utf-8")

    print(f"[app] /segment-image done after {time.time() - started_at:.1f}s: {result['mask_path']}", flush=True)
    return jsonify({
        "status": "ok",
        "mask_path": result["mask_path"],
        "mask_url": public_file_url(result["mask_path"]),
        "mask_base64": result.get("mask_base64"),
        "score": result.get("score"),
        "selected_mask_index": result.get("selected_mask_index"),
        "selection_mode": result.get("selection_mode"),
        "image_width": result.get("image_width"),
        "image_height": result.get("image_height"),
    })


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("PORT", "8081"))
    auto_shutdown_seconds = int(os.getenv("SERVER_AUTO_SHUTDOWN_SECONDS", "0"))
    _schedule_auto_shutdown(auto_shutdown_seconds)
    app.run(host="0.0.0.0", port=port, debug=debug)
