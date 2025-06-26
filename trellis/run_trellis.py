#!/usr/bin/env python3
# trellis/run_trellis.py  – single-file wrapper to run TRELLIS and save *all* outputs
#
# Usage examples
#   python trellis/run_trellis.py "A wooden toy truck"
#   python trellis/run_trellis.py ./inputs/photo.png
#   python trellis/run_trellis.py ./inputs/multi_view_folder/
#
# Output hierarchy (created in CWD):
#   outputs/
#       sample_gaussian.ply
#       sample_mesh.obj           (high-poly mesh)
#       sample.glb                (simplified + textured)
#       sample_rf_mesh.obj        (mesh marched from radiance-field)
#       sample_gs.mp4             (turn-table video of Gaussians)

import os, re, sys, argparse, base64, json
from io import BytesIO
from pathlib import Path

os.environ["SPCONV_ALGO"] = "native"      # safer first-run
sys.path.append("/usr/local/lib/python3.10/site-packages")

import openai, imageio
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# ───────────── CLI ─────────────
parser = argparse.ArgumentParser(description="TRELLIS X-to-3D wrapper")
parser.add_argument("input", help="text prompt, image.png or folder/")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--outdir", default="outputs")
args = parser.parse_args()

outdir = Path(args.outdir).resolve()
outdir.mkdir(parents=True, exist_ok=True)

# ──────────── Pipeline ─────────
pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "gqk/TRELLIS-image-large-fork" # or "JeffreyXiang/TRELLIS-image-large"
).cuda()

# ──────── Prepare input image(s) ────────
img_list: list[Image.Image] = []

if args.input.lower().endswith(".png"):
    img_list.append(Image.open(args.input))

elif Path(args.input).is_dir():
    for p in sorted(Path(args.input).glob("*.png")):
        img_list.append(Image.open(p))
    if not img_list:
        sys.exit("No .png files found in folder")

else:  # assume pure text -> use DALL-E 3 to get a single conditioning image
    client = openai.OpenAI()
    resp = client.images.generate(
        model="dall-e-3",
        prompt=args.input,
        size="1024x1024",
        quality="standard",
        response_format="b64_json",
        n=1,
    )
    img_data = resp.data[0].model_dump()["b64_json"]
    img = Image.open(BytesIO(base64.b64decode(img_data)))
    img.save(outdir / "prompt_image.png")
    img_list.append(img)

# ─────────── Run TRELLIS ───────────
if len(img_list) > 1:
    outputs = pipeline.run_multi_image(img_list, 
    seed=args.seed,
    formats=["gaussian", "mesh", "radiance_field"]
    )
else:
    outputs = pipeline.run(img_list[0], 
    seed=args.seed,
    formats=["gaussian", "mesh", "radiance_field"]
    )

gaussian   = outputs["gaussian"][0]
mesh       = outputs["mesh"][0]
radiance_f = outputs["radiance_field"][0]

# ─────────── Save Gaussian PLY ───────────
ply_path = outdir / "sample_gaussian.ply"
gaussian.save_ply(ply_path)
print(f"Gaussian PLY  -> {ply_path}")

# ─────────── Save high-poly mesh ─────────
obj_path = outdir / "sample_mesh.obj"
mesh.export(obj_path)
print(f"Mesh OBJ      -> {obj_path}")

# ─────────── Simplified textured GLB ─────
glb_path = outdir / "sample.glb"
glb = postprocessing_utils.to_glb(
    gaussian,
    mesh,
    # Optional parameters
    simplify=0.95,     # keep 5 % faces: Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)
glb.export(glb_path)
print(f"GLB (textured) -> {glb_path}")

# ─────────── Radiance-Field -> Mesh (optional) ─────────
rf_mesh_path = outdir / "sample_rf_mesh.obj"
rf_mesh = postprocessing_utils.rf_to_mesh(radiance_f, density_thresh=50.0)
rf_mesh.export(rf_mesh_path)
print(f"RF mesh       -> {rf_mesh_path}")

# ─────────── Turn-table render of Gaussians ─────────
video_path = outdir / "sample_gs.mp4"
video = render_utils.render_video(gaussian)["color"]
imageio.mimsave(video_path, video, fps=30)
print(f"Preview video -> {video_path}")
