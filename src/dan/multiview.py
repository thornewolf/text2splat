"""Generate multi-view images of a subject using the Pixverse image template API.

Workflow: upload image → submit N generation jobs (one template per angle) →
poll until complete → download and save outputs.

Template IDs must be obtained from the Pixverse Effect Center
(https://app.pixverse.ai/effect). Pick templates that produce a specific
camera viewpoint or perspective shift — look for "multi-view", "3D rotate",
or similar categories.
"""

import os
import time
import uuid
from pathlib import Path

import requests

PIXVERSE_BASE = "https://app-api.pixverse.ai/openapi/v2"

# Status codes returned by the result endpoint
_STATUS_COMPLETE = 1
_STATUS_PROCESSING = 5

# Default template IDs for 5 distinct camera angles.
# Replace these with real IDs from the Pixverse Effect Center.
# The example value 377378608924544 comes from the official docs.
DEFAULT_TEMPLATE_IDS: list[int] = [
    377378608924544,  # view 0 — update from Effect Center
    377378608924544,  # view 1 — update from Effect Center
    377378608924544,  # view 2 — update from Effect Center
    377378608924544,  # view 3 — update from Effect Center
    377378608924544,  # view 4 — update from Effect Center
]

ANGLE_LABELS = ["front", "left_45", "right_45", "rear", "top"]


def _trace_id() -> str:
    return str(uuid.uuid4())


def _headers(api_key: str) -> dict[str, str]:
    return {"API-KEY": api_key, "Ai-Trace-Id": _trace_id()}


def _upload_image(image_path: str, api_key: str) -> int:
    path = Path(image_path)
    ext = path.suffix.lower().lstrip(".")
    mime_map = {"webp": "image/webp", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime = mime_map.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        resp = requests.post(
            f"{PIXVERSE_BASE}/image/upload",
            headers=_headers(api_key),
            files={"image": (path.name, f, mime)},
        )
    resp.raise_for_status()
    data = resp.json()
    if data["ErrCode"] != 0:
        raise RuntimeError(f"Upload failed [{data['ErrCode']}]: {data['ErrMsg']}")
    return data["Resp"]["img_id"]


def _submit_job(img_id: int, template_id: int, api_key: str) -> int:
    resp = requests.post(
        f"{PIXVERSE_BASE}/image/template/generate",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json={"img_ids": [img_id], "template_id": template_id},
    )
    resp.raise_for_status()
    data = resp.json()
    if data["ErrCode"] != 0:
        raise RuntimeError(f"Generation submit failed [{data['ErrCode']}]: {data['ErrMsg']}")
    return data["Resp"]["image_id"]


def _poll_result(image_id: int, api_key: str, timeout: int = 180, poll_interval: int = 5) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(
            f"{PIXVERSE_BASE}/image/result/{image_id}",
            headers=_headers(api_key),
        )
        resp.raise_for_status()
        data = resp.json()
        if data["ErrCode"] != 0:
            raise RuntimeError(f"Result error [{data['ErrCode']}]: {data['ErrMsg']}")
        result = data["Resp"]
        status = result.get("status")
        if status == _STATUS_COMPLETE:
            url = result.get("url")
            if not url:
                raise RuntimeError(f"Job {image_id} complete but no URL in response")
            return url
        if status != _STATUS_PROCESSING:
            raise RuntimeError(f"Job {image_id} unexpected status: {status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {image_id} timed out after {timeout}s")


def _download(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def generate_multiview(
    image_path: str,
    num_views: int = 3,
    output_dir: str | None = None,
    api_key: str | None = None,
    template_ids: list[int] | None = None,
    poll_timeout: int = 180,
) -> list[str]:
    """Generate ``num_views`` images of the subject from different camera angles.

    Args:
        image_path: Path to the input image (PNG, JPEG, or WebP, max 20 MB).
        num_views: Number of distinct views to generate (default 3, max 5).
        output_dir: Directory to write output images.  Defaults to
            ``<this_file's_directory>/output``.
        api_key: Pixverse API key.  Falls back to the ``PIXVERSE_API_KEY``
            environment variable.
        template_ids: Pixverse template IDs — one per view.  Must contain at
            least ``num_views`` entries.  Defaults to ``DEFAULT_TEMPLATE_IDS``;
            update those with real IDs from the Pixverse Effect Center.
        poll_timeout: Seconds to wait per job before raising ``TimeoutError``.

    Returns:
        Sorted list of absolute paths to the saved output images.
    """
    api_key = api_key or os.environ.get("PIXVERSE_API_KEY")
    if not api_key:
        raise ValueError("No API key — set the PIXVERSE_API_KEY environment variable.")

    if template_ids is None:
        template_ids = DEFAULT_TEMPLATE_IDS
    if len(template_ids) < num_views:
        raise ValueError(
            f"Need at least {num_views} template_ids but got {len(template_ids)}."
        )

    src_dir = Path(__file__).parent
    out_dir = Path(output_dir) if output_dir else src_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    suffix = Path(image_path).suffix or ".webp"

    # --- 1. Upload the source image once ---
    print(f"Uploading {image_path} …")
    img_id = _upload_image(image_path, api_key)
    print(f"  img_id = {img_id}")

    # --- 2. Submit all generation jobs up front ---
    jobs: list[tuple[int, str, int]] = []  # (job_image_id, label, view_index)
    for i in range(num_views):
        template_id = template_ids[i]
        label = ANGLE_LABELS[i] if i < len(ANGLE_LABELS) else f"view{i}"
        print(f"  Submitting view {i + 1}/{num_views}: {label} (template {template_id}) …")
        job_id = _submit_job(img_id, template_id, api_key)
        jobs.append((job_id, label, i))

    # --- 3. Poll each job and download the result ---
    output_paths: list[str] = []
    for job_id, label, i in jobs:
        print(f"  Waiting for view {i + 1}/{num_views}: {label} (job {job_id}) …")
        url = _poll_result(job_id, api_key, timeout=poll_timeout)
        dest = out_dir / f"{stem}_{label}{suffix}"
        _download(url, dest)
        print(f"  Saved → {dest}")
        output_paths.append(str(dest.resolve()))

    print(f"Done. {len(output_paths)} images saved to {out_dir}")
    return output_paths


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "src/dan/input/quokka.webp"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    results = generate_multiview(input_path, num_views=num)
    for p in results:
        print(p)
