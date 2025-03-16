# Import OS libaries and define backend 
import os
import re
import sys

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.


# Import libraries for text-to-2D (via DALL-E)
import openai
import base64
from io import BytesIO

# Import libaries for 2D-to-3D (via TRELLIS) 
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Import libraries for argument parsing
import argparse

# Parse input arguments
parser = argparse.ArgumentParser(
        prog="X-to-3D Pipeline example",
        description="This code takes in an input, evaluates an LLM, and generates a 3D model as the output. It is currently configured to accept either text (which activates the text-to-3D pipeline) or image (which activates the image-to-3D pipeline) as input."
        )
parser.add_argument("text_input")
args = parser.parse_args()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Initialize image list
image = []

# Input option 1: Single image (activates image-to-3D pipeline)
if re.search(".png$",args.text_input):
    image.append(Image.open(args.text_input))
    print('Found png image as input. Activating the image-to-3D pipeline.')

# Input option 2: Multi-image (activates multi-image-to-3D pipeline)
elif os.path.isdir(args.text_input):
    print('Found directory as input. Looking for images in directory...')   
    for filename in os.listdir(args.text_input): 
        if re.search(".png$",filename):
            image.append(Image.open(args.text_input+'/'+filename))
    if len(image) > 0:
        print('Found ' + str(len(image)) + ' images. Activating the image-to-3D pipeline.')  
    else:
        print('No images found. Exiting...')
        sys.exit("No PNG images found in input directory.")

# Input option 3: Text (activates the text-to-3D pipeline)
else:
    client = openai.OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=args.text_input,
        size="1024x1024",
        quality="standard",
        response_format="b64_json",
        n=1)
    img_data = response.data[0]
    img_obj = img_data.model_dump()["b64_json"]
    img_buffer = BytesIO(base64.b64decode(img_obj))
    image.append(Image.open(img_buffer))
    image[0].save('./object2D.png')
    print('Found text string as input. Activating the text-to-3D pipeline.')


# Run the pipeline
if len(image) > 1:
    outputs = pipeline.run_multi_image(
        image,
        seed=1,
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
else:
    outputs = pipeline.run(
        image[0],
        seed=1,
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )


# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
save_directory = os.path.join(os.getcwd(), "output")
os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(save_directory+"/sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave(save_directory+"/sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(save_directory+"/sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(save_directory+"/sample.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(save_directory+"/sample.ply")
