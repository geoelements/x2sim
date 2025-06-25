#!/usr/bin/env python3
# x2sim/x2sim.py
import os
import sys
import subprocess
import json
import time
import argparse
import logging
import shutil
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools.base import BaseTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("x2sim/x2sim.log"), logging.StreamHandler()]
)
logger = logging.getLogger("x2sim")





# Custom tool to generate a point cloud from text
class TextToPointCloudTool(BaseTool):
    name = "TextToPointCloud"
    description = "Generates a 3D point cloud from a given text object description using TRELLIS to be used in a simulation later."

    def _run(self, prompt: str) -> str:
        """Generate a TRELLIS point cloud based on the given prompt."""
        logger.info(f"Generating 3D model for prompt: '{prompt}'")
        
        try:
            # Save the prompt to a temporary file for the shell script to read
            with open("outputs/current_prompt.txt", "w") as f:
                f.write(prompt)
            
            # Run the script
            result = subprocess.run(["python3","trellis/run_trellis.py",prompt], 
                                   shell=False, 
                                   check=True,
                                   capture_output=True,
                                   text=True)
            
            logger.info("3D model generation completed")
            
            # Check if output file exists - ONLY check the current working directory
            output_file = os.path.join(os.getcwd(), "outputs/sample_gaussian.ply")
            if not os.path.exists(output_file):
                logger.error(f"3D model file not found at: {output_file}")
                return f"Error: 3D model file not found at {output_file}"

            logger.info(f"Found 3D model at: {output_file}")
                        
            # Copy or link the file to preprocessed.ply in the current directory for MPM simulation
            destination_file = os.path.join(os.getcwd(), "outputs/preprocessed.ply")
            subprocess.run(["cp", output_file, destination_file], check=True)
            logger.info(f"Copied 3D model to {destination_file}")
            
            return f"TRELLIS point cloud generated successfully. File saved to {destination_file}"
            
        except Exception as e:
            logger.error(f"3D model generation failed: {e}")
            return f"Error generating TRELLIS point cloud: {str(e)}"

    async def _arun(self, prompt: str) -> str:
        """Asynchronous version, not implemented."""
        raise NotImplementedError("TrellisGeneration does not support async")






# Custom tool to generate a point cloud from image(s)
class ImageToPointCloudTool(BaseTool):
    name = "ImageToPointCloud"
    description = "Processes one or more images to create a 3D point cloud for simulation. Can handle a single image file or a directory of images."

    def _run(self, image_path: str) -> str:
        """
        Process a single image or directory of images to generate a point cloud.
        
        Args:
            image_path (str): Path to an image file (.png) or directory containing images
        """
        logger.info(f"Processing image to point cloud: {image_path}")
        
        try:
            # Check if input is a valid path
            if not os.path.exists(image_path):
                logger.error(f"Image path not found: {image_path}")
                return f"Error: Image path not found: {image_path}"
            
            # Run the TRELLIS script to process the image(s)
            logger.info(f"Running TRELLIS on image path: {image_path}")
            
            # Run the script with the image path
            result = subprocess.run(
                ["python3", "trellis/run_trellis.py", image_path],
                shell=False,
                check=False,
                capture_output=True,
                text=True
            )
            
            # Check if the script execution was successful
            if result.returncode != 0:
                logger.error(f"TRELLIS processing failed: {result.stderr}")
                return f"Error: TRELLIS processing failed. Check logs for details."
            
            # Check if output file exists - First check the outputs directory
            output_file = os.path.join(os.getcwd(), "outputs/sample_gaussian.ply")
            if not os.path.exists(output_file):
                logger.error(f"3D model file not found at: {output_file}")
                return f"Error: 3D model file not found at {output_file}"
            
            # Copy or link the file to preprocessed.ply in the outputs directory for MPM simulation
            destination_file = os.path.join(os.getcwd(), "outputs/preprocessed.ply")
            subprocess.run(["cp", output_file, destination_file], check=True)
            logger.info(f"Copied 3D model to {destination_file}")
            
            # Determine if we processed a single image or multiple images
            if os.path.isdir(image_path):
                image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
                return f"Successfully processed {len(image_files)} images to create a 3D point cloud. File saved to {destination_file}"
            else:
                return f"Successfully processed image to create a 3D point cloud. File saved to {destination_file}"
            
        except Exception as e:
            logger.error(f"Error in image to point cloud processing: {str(e)}")
            return f"Error processing image: {str(e)}"

    async def _arun(self, image_path: str) -> str:
        """Asynchronous version, not implemented."""
        raise NotImplementedError("ImageToPointCloud does not support async")




    

# Custom tool to generate a point cloud from a video
class VideoToPointCloudTool(BaseTool):
    name = "VideoToPointCloud"
    description = "Processes a YouTube video URL or local video file to extract frames and create a 3D point cloud for simulation."

    def _run(self, video_url: str) -> str:
        """
        Process a video URL to extract frames and generate a point cloud.
        
        Args:
            video_url (str): YouTube URL or local video path
        """
        logger.info(f"Processing video to point cloud: {video_url}")
        
        try:
            # Import the video frame extraction utility
            try:
                from video_utils import extract_video_frames
                logger.info("Successfully imported extract_video_frames from video_utils module")
            except ImportError:
                logger.error("Failed to import video_utils module. Make sure video_utils.py is in your working directory.")
                return "Error: Required video_utils module is missing."
            
            # Extract frames from the video
            frames_dir = extract_video_frames(video_url)
            if not frames_dir:
                return "Error: Failed to extract frames from the video."
            
            # Check if there are frames in the directory
            frame_files = [f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')]
            if not frame_files:
                logger.error(f"No frames were extracted to {frames_dir}")
                return "Error: No frames were extracted from the video."
            
            logger.info(f"Successfully extracted {len(frame_files)} frames to {frames_dir}")
            
            # Process the frames using the videoto3Dexample.sh script
            logger.info(f"Running TRELLIS on frames directory: {frames_dir}")
            
            # Run the script with the frames directory
            result = subprocess.run(
                ["python3","trellis/run_trellis.py",frames_dir],
                shell=False,  # Don't use shell=True with array arguments
                check=False,  # Don't raise exception, handle errors manually
                capture_output=True,
                text=True
            )

            # Copy or link the file to preprocessed.ply in the current directory for MPM simulation
            output_file = os.path.join(os.getcwd(), "outputs/sample_gaussian.ply")
            if not os.path.exists(output_file):
                logger.error(f"3D model file not found at: {output_file}")
                return f"Error: 3D model file not found at {output_file}"
            destination_file = os.path.join(os.getcwd(), "outputs/preprocessed.ply")
            subprocess.run(["cp", output_file, destination_file], check=True)
            logger.info(f"Copied 3D model to {destination_file}")
            
            # Check if the script execution was successful
            if result.returncode != 0:
                logger.error(f"TRELLIS processing failed: {result.stderr}")
                return f"Error: TRELLIS processing failed. Check logs for details."
            
            # Check if the output file exists
            output_file = os.path.join(os.getcwd(), "outputs/preprocessed.ply")
            if not os.path.exists(output_file):
                logger.error("Output file not found after TRELLIS processing")
                return "Error: Output file not found after TRELLIS processing"
            
            logger.info(f"Video processed successfully, 3D model saved to: {output_file}")
            return f"Video processed successfully. 3D point cloud saved to {output_file}"
            
        except Exception as e:
            logger.error(f"Error in video to point cloud processing: {str(e)}")
            return f"Error processing video: {str(e)}"

    async def _arun(self, video_url: str) -> str:
        """Asynchronous version, not implemented."""
        raise NotImplementedError("VideoToPointCloud does not support async")




# Custom tool to generate fluid simulation parameters
class FluidSimulationTool(BaseTool):
    name = "FluidSimulation"
    description = "Generates water to be used in MPM simulation later"

    def _run(self, input_data: str) -> str:
        """Generate fluid simulation parameters."""
        logger.info(f"Generating fluid simulation with direction: '{input_data}'")
        
        try:
            # Import the fluid generation function
            sys.path.append(os.getcwd())
            sys.path.append(os.getcwd()+'/x2sim')
            try:
                from fluid_sim_agent import gen_water
            except ImportError:
                logger.warning("Could not import gen_water from fluid_sim_agent, trying fluidgen")
                from fluidgen import gen_fluid
            
            result = gen_fluid(input_data)
            logger.info("Fluid simulation parameters generated")
            
            # Check if the JSON file was created
            fluid_json_path = os.path.join(os.getcwd(), "x2sim/fluid_simulation.json")
            if not os.path.exists(fluid_json_path):
                logger.error("Fluid simulation JSON file not created")
                return "Error: Fluid simulation JSON file not created"
                
            return f"Water direction generated successfully. Parameters saved to {fluid_json_path}"
            
        except Exception as e:
            logger.error(f"Fluid simulation generation failed: {e}")
            return f"Unexpected error in generating water direction: {str(e)}"

    async def _arun(self, input_data: str) -> str:
        """Asynchronous version, not implemented."""
        raise NotImplementedError("FluidSimulation does not support async")






# Custom tool to run MPM simulations
class MPMSimulationTool(BaseTool):
    name = "MPMSimulation"
    description = "Runs an MPM (Material Point Method) simulation using a previously generated point cloud and water direction."

    def _run(self, input_data: str) -> str:
        """Run MPM simulation."""
        logger.info("Running MPM simulation")
        
        try:
            # Import the MPM simulation function
            sys.path.append(os.getcwd())
            sys.path.append(os.getcwd()+'/x2sim')
            try:
                from mpm_sim import gen_sim
            except ImportError:
                logger.warning("Could not import gen_sim from mpm_sim, trying mpmsim")
                from mpmsim import gen_sim
            
            result = gen_sim(input_data)
            logger.info("MPM simulation completed")
            
            # Check for simulation output
            sim_output_dir = os.path.join(os.getcwd(), "outputs")
            if not os.path.exists(sim_output_dir):
                logger.error("Simulation output directory not found")
                return "Error: Simulation output directory not found"
                
            return f"MPM simulation completed successfully. Results saved to {sim_output_dir}"
            
        except Exception as e:
            logger.error(f"MPM simulation failed: {e}")
            return f"Unexpected error in MPM simulation: {str(e)}"

    async def _arun(self, input_data: str) -> str:
        """Asynchronous version, not implemented."""
        raise NotImplementedError("MPMSimulation does not support async")






def setup_agent():
    """Setup the LangChain agent with tools"""
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not available for agent setup")
        return None
    
    # Define the LLM
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-4o")
    
    # Create tool instances
    text_to_pc_tool = TextToPointCloudTool()
    video_pc_tool = VideoToPointCloudTool()
    image_pc_tool = ImageToPointCloudTool()  # Add new image tool
    fluid_simulation_tool = FluidSimulationTool()
    mpm_sim_tool = MPMSimulationTool()
    
    # Define tools available to the agent
    tools = [text_to_pc_tool, video_pc_tool, image_pc_tool, fluid_simulation_tool, mpm_sim_tool]  # Add image tool
    
    # Create the REACT agent's prompt
    agent_prompt = PromptTemplate.from_template(
        "You are an assistant capable of running three types of simulations: Text-to-3D, Video-to-3D, and Image-to-3D simulations. " +
        "You have access to the following tools:\n" +
        "{tool_names}\n\n" +
        "Here are the descriptions of the tools:\n" +
        "{tools}\n\n" +
        "Based on the user's input, decide which simulation to run. " +
        "If any of the scripts fail, mention there was an error in the script.\n" +
        "If the input mentions generating 3D objects or scenes without mentioning a video or image, ALWAYS use the TextToPointCloud tool. " +
        "If the input specifically mentions processing a video, use the VideoToPointCloud tool.\n" +
        "If the input specifically mentions processing an image or images, use the ImageToPointCloud tool.\n\n" +
        "CRITICAL INSTRUCTION: After successfully using either the TextToPointCloud, VideoToPointCloud, or ImageToPointCloud tool, " +
        "you MUST ALWAYS use the FluidSimulation tool and MPMSimulation tool sequentially on the generated point cloud. This is a mandatory step. " +
        "You can use these tools to help answer the following question: {input}\n\n" +
        "To use a tool, please use the following format:\n" +
        "Thought: Consider what to do next\n" +
        "Action: Tool name\n" +
        "Action Input: Input to the tool\n" +
        "Observation: Result of the tool\n\n" +
        "When you have a final answer, respond with:\n" +
        "Thought: I have the final answer\n" +
        "Final Answer: Your final answer here\n\n" +
        "IMPORTANT: When using the TextToPointCloud tool, the Action Input should be a brief description of the object or scene, NOT the entire user input. " +
        "For example:\n" +
        "User Input: 'Generate a blue truck to be used in a simulation later.'\n" +
        "Correct Action Input: 'A blue truck'\n\n" +
        "Always extract just the object or scene description for the TextToPointCloud tool.\n" +
        "IMPORTANT: When using the ImageToPointCloud tool, the Action Input should be the path to the image file or directory of images.\n" +
        "IMPORTANT: When using the FluidSimulation tool, the Action Input should be brief description of the water direction, NOT the entire user input. "+
        "For example:\n" +
        "User Input: 'Generate a tall building being hit with water from top'" +
        "Correct Action Input: 'generate water when it is hitting the object from its top'\n\n" +
        "Another example:\n" +
        "User Input: 'Generate a tall building being hit with wave from its front'" +
        "Correct Action Input: 'generate wave when it is hitting the object from its front'\n\n" +
        "Always describe the direction of water from the {input} for the FluidSimulation tool.\n" +
        "Remember to ALWAYS use the MPMSimulation tool at the end.\n" +
        "{agent_scratchpad}"
    )
    
    # Create the agent using the prompt
    agent = create_react_agent(tools=tools, llm=llm, prompt=agent_prompt)
    logger.info("Agent created successfully")
    
    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor






def run_agent_pipeline(input_data, use_video=False, video_url=None, use_image=False, image_path=None):
    """Run the agent-based pipeline
    
    Args:
        input_data (str): The prompt or description for the simulation
        use_video (bool): Whether to use video processing instead of text-to-3D
        video_url (str, optional): URL or path of the video to process if use_video is True
        use_image (bool): Whether to use image processing instead of text-to-3D
        image_path (str, optional): Path to image file or directory of images if use_image is True
    
    Returns:
        bool: Whether the pipeline execution was successful
    """
    logger.info(f"Starting agent-based pipeline with input: '{input_data}'")
    
    # Setup the agent
    agent_executor = setup_agent()
    if not agent_executor:
        logger.error("Failed to setup agent executor")
        return False
    
    # Handle YouTube URLs if specified
    video_path = None
    if use_video and video_url:
        if "youtube.com" in video_url or "youtu.be" in video_url:
            logger.info(f"Detected YouTube URL: {video_url}")
            try:
                # Import the video_utils module for downloading YouTube videos
                from video_utils import download_youtube_video
                video_path = download_youtube_video(video_url)
                if not video_path:
                    logger.error("Failed to download YouTube video")
                    return False
            except ImportError:
                logger.error("Failed to import download_youtube_video from video_utils")
                return False
        else:
            # Assume it's a local file path
            video_path = video_url
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
    
    # Handle image path if specified
    if use_image and image_path:
        if not os.path.exists(image_path):
            logger.error(f"Image path not found: {image_path}")
            return False
    
    # Prepare the prompt based on the mode
    if use_video and video_path:
        prompt = f"Process this video: {video_path} and simulate it with {input_data}"
    elif use_image and image_path:
        if os.path.isdir(image_path):
            prompt = f"Process these images in directory: {image_path} and simulate it with {input_data}"
        else:
            prompt = f"Process this image: {image_path} and simulate it with {input_data}"
    else:
        prompt = input_data
    
    # Prepare the input dictionary
    agent_input = {
        "input": prompt,
        "tool_names": ", ".join([tool.name for tool in agent_executor.tools]),
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in agent_executor.tools]),
        "agent_scratchpad": ""
    }
    
    # Invoke the agent with the dictionary
    try:
        result = agent_executor.invoke(agent_input, return_intermediate_steps=True)
        logger.info("Agent pipeline completed successfully")
        logger.debug(f"Agent result: {result}")
        return True
    except Exception as e:
        logger.error(f"Agent pipeline failed: {e}")
        return False








def main():
    parser = argparse.ArgumentParser(description="x2sim - Text/Video/Image to 3D to Simulation Pipeline with Agent-Based Approach")
    # Create a group for input source (text, video, or image)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--prompt", type=str, 
                        help="Text prompt describing the object and fluid simulation")
    input_group.add_argument("--video", type=str,
                        help="URL or path to a video file to process")
    input_group.add_argument("--image", type=str,
                        help="Path to an image file (.png) or directory containing image files to process")
    
    # Fluid direction argument
    parser.add_argument("--direction", type=str, default="from the front",
                        help="Direction of the fluid relative to the object (e.g., 'from the top', 'from the front')")
    
    # Other optional arguments
    parser.add_argument("--use-venv", action="store_true", 
                        help="Virtual environment activation")
    parser.add_argument("--object", type=str,
                        help="Override the object description (used with --video or --image)")
    
    args = parser.parse_args()
    
    # Handle the case where no arguments are provided
    if not args.prompt and not args.video and not args.image:
        parser.print_help()
        sys.exit(1)
    
    # Prepare the input for the agent pipeline
    use_video = bool(args.video)
    use_image = bool(args.image)
    
    # Determine the prompt
    if args.prompt:
        input_text = args.prompt
    elif args.video and args.object:
        input_text = f"Generate a {args.object} being hit with water {args.direction}"
    elif args.video:
        input_text = f"Generate an object being hit with water {args.direction}"
    elif args.image and args.object:
        input_text = f"Generate a {args.object} being hit with water {args.direction}"
    elif args.image:
        input_text = f"Generate an object being hit with water {args.direction}"
    else:
        input_text = "Generate a house being hit with a wave from the front"
    
    # Run the agent-based pipeline
    success = run_agent_pipeline(input_text, use_video, args.video, use_image, args.image)
    
    if success:
        logger.info("Pipeline executed successfully")
        sys.exit(0)
    else:
        logger.error("Pipeline execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
