# text2splat

## Multi-Image Gaussian Splat

Use `scripts/run_multiview_splat.sh` when you have multiple photos of the same
object from different angles. It uses Nerfstudio's COLMAP-based image processing,
trains `splatfacto`, exports a `.ply` Gaussian splat, and renders an `.mp4`.

First-time setup on macOS usually needs:

```bash
brew install colmap ffmpeg pkg-config
```

Nerfstudio's `splatfacto` training path currently requires CUDA. On Apple
Silicon, the script can process images with COLMAP, but multi-image Gaussian
splat training needs to run on a CUDA Linux machine.

Then run:

```bash
scripts/run_multiview_splat.sh \
  -i /Users/thorn/Downloads/object_photos \
  --install-env \
  --max-num-iterations 7000
```

If `/Users/thorn/code/nerfstudio` exists, `--install-env` installs that clone
editable. Otherwise it falls back to the PyPI package. To force a specific clone:

```bash
scripts/run_multiview_splat.sh \
  -i /Users/thorn/Downloads/object_photos \
  --install-env \
  --nerfstudio-source /Users/thorn/code/nerfstudio
```

Outputs are written under `runs/multiview_splat/<run-name>/`:

```text
processed/      # Nerfstudio/COLMAP processed data
nerfstudio/     # training run and config.yml
export/         # exported .ply Gaussian splat
renders/        # rendered .mp4
```

For a quick command preview without running training:

```bash
scripts/run_multiview_splat.sh \
  -i /Users/thorn/Downloads/object_photos \
  --max-num-iterations 10 \
  --dry-run
```

The default render uses `ns-render spiral`. For a better controlled orbit, open
the trained run in the viewer, create/export a camera path JSON, then render
from the existing config:

```bash
scripts/run_multiview_splat.sh \
  --load-config runs/multiview_splat/<run-name>/nerfstudio/<run-name>/splatfacto/<timestamp>/config.yml \
  --render-mode camera-path \
  --camera-path /Users/thorn/Downloads/camera_path.json
```

## TSW SHARP Trial

The `tsw` package includes a thin trial wrapper around Apple's SHARP project:

```bash
uv run tsw-sharp -i /path/to/image-or-directory -o /path/to/output-dir
```

By default, this delegates to:

```bash
uvx --from git+https://github.com/apple/ml-sharp sharp predict
```

SHARP downloads its model checkpoint on first use and writes `.ply` Gaussian splat
files to the output directory. Useful options:

```bash
uv run tsw-sharp -i input.jpg -o outputs --device mps
uv run tsw-sharp -i input.jpg -o outputs --device cpu --dry-run
uv run tsw-sharp -i input.jpg -o outputs -c sharp_2572gikvuh.pt
```
