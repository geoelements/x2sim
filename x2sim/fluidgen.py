#!/usr/bin/env python3
"""
Fluid simulation agent for generating fluid parameters based on directional input
"""
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools.base import BaseTool
from dotenv import load_dotenv
import subprocess
import os
import json
import ast
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("x2sim/fluid_sim.log"), logging.StreamHandler()]
)
logger = logging.getLogger("fluid_sim")

def gen_fluid(input_data):
    """Generate fluid simulation parameters from text input"""
    # Load the environment variables from the .env file
    #load_dotenv(dotenv_path='./openai_api_key.env')

    # Access the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError("OPENAI_API_KEY is required for fluid simulation")

    # Define the LLM
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-4")

    class GeneratePositionTool(BaseTool):
        name = "GeneratePosition"
        description = "Generates initial position of the fluid cube based on the input array"

        def _run(self, input: str) -> str:
            """Save initial position based on the input array"""
            logger.info(f"Generating initial position for fluid: {input}")
            try:
                # Try to parse the input as an array
                position_array = ast.literal_eval(input)
                
                # Validate the position
                if len(position_array) != 6:
                    return f"Error: Position array must have 6 elements, got {len(position_array)}"
                
                # Check bounds
                for i in range(3):
                    if position_array[i] < 0.1 or position_array[i] > 1.9:
                        return f"Error: Position coordinate {i} out of bounds: {position_array[i]}"
                    if position_array[i] + position_array[i+3] > 1.9:
                        return f"Error: Position + size exceeds boundary at dimension {i}"
                
                # Create the data structure
                data = {"cubes": [position_array]}
                
                with open("x2sim/fluid_simulation.json", "w") as json_file:
                    json.dump(data, json_file, indent=4)

                return f"Fluid cube position generated successfully: {position_array}"
            
            except Exception as e:
                logger.error(f"Error generating fluid cube position: {str(e)}")
                return f"Error generating fluid cubes position: {str(e)}"

        async def _arun(self, input_video: str) -> str:
            """Asynchronous version, not implemented."""
            raise NotImplementedError("GeneratePosition does not support async")
        
    class GenerateVelocityTool(BaseTool):
        name = "GenerateVelocity"
        description = "Generates initial velocity for the generated fluid based on the input array"

        def _run(self, input: str) -> str:
            """Save initial velocity based on the input array"""
            logger.info(f"Generating initial velocity for fluid: {input}")
            
            try:
                # Try to parse the input as an array
                velocity_array = ast.literal_eval(input)
                
                # Validate the velocity
                if len(velocity_array) != 3:
                    return f"Error: Velocity array must have 3 elements, got {len(velocity_array)}"
                
                # Check velocity ranges
                for i in range(3):
                    if abs(velocity_array[i]) > 6:
                        return f"Error: Velocity component {i} exceeds maximum allowed: {velocity_array[i]}"
                
                # Read the existing JSON
                if not os.path.exists("x2sim/fluid_simulation.json"):
                    return "Error: fluid_simulation.json does not exist. Run GeneratePosition first."
                    
                with open("x2sim/fluid_simulation.json", "r") as json_file:
                    data = json.load(json_file)
                
                # Add the velocity data
                data["velocity"] = [velocity_array]
                
                # Write back the updated JSON
                with open("x2sim/fluid_simulation.json", "w") as json_file:
                    json.dump(data, json_file, indent=4)

                return f"Fluid cube velocity generated successfully: {velocity_array}"
            
            except Exception as e:
                logger.error(f"Error generating fluid cube velocity: {str(e)}")
                return f"Error generating fluid cubes velocity: {str(e)}"

        async def _arun(self, input_data: str) -> str:
            """Asynchronous version, not implemented."""
            raise NotImplementedError("GenerateVelocity does not support async")
        
    # Create tool instances
    generate_position_tool = GeneratePositionTool()
    generate_velocity_tool = GenerateVelocityTool()

    # Define tools available to the agent
    tools = [generate_position_tool, generate_velocity_tool]

    # Create the REACT agent's prompt
    agent_prompt = PromptTemplate.from_template(
        "You are an assistant capable of determining the initial position and initial velocity of a fluid cuboid in a 3D space based on {input}. " +
        "The 3D space has bounds [[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]], where the first index is the origin and the second index represents the lengths along the x, y, and z axes. " +
        "The x and z axes are at the bottom plane, and the y axis is the vertical direction. " +
        "There is an object positioned at the center of the bottom xz-plane, symmetrically along the x-axis, and oriented along the z-axis, with its front at lower z-values.\n " + 
        "you have to choose initial position and velocity such that it hits that object.\n" +
        "You have access to the following tools:\n" +
        "{tool_names}\n\n" +
        "Here are the descriptions of the tools:\n" +
        "{tools}\n\n" +
        "You MUST ALWAYS use GeneratePositionTool and GenerateVelocityTool sequentially" +
        "If any of the scripts fail, mention there was an error in the script.\n" +
        "You can use these tools to help answer the following question: {input}\n\n" +
        "To use a tool, please use the following format:\n" +
        "Thought: Consider what to do next\n" +
        "Action: Tool name\n" +
        "Action Input: Input to the tool\n" +
        "Observation: Result of the tool\n\n" +
        "When you have a final answer, respond with:\n" +
        "Thought: I have the final answer\n" +
        "Final Answer: Your final answer here\n\n" +
        "IMPORTANT: When using the GeneratePosition tool, the format for the action input must be an array [x1, y1, z1, lx, ly, lz], where x1, y1, z1 are the coordinates of the bottom-left corner of the cuboid, and lx, ly, lz are the lengths along the x, y, and z axes.\n" +
        "The lengths lx, ly and lz must be between 0.4 to 0.8 except for a wave and they need not to be equal\n" +
        "CRITICAL INSTRUCTION: The sums (x1 + lx), (y1 + ly) and (z1 + lz) should never be greater than 1.9\n" +
        "The position of the fluid cuboid must be the one that best describes the {input}" +
        "EXCEPTION: If a wave is mentioned, set the fluid's y-bound to (0.1, 1). The dimension perpendicular to the flow should be (0.1, 1.5), and the dimension parallel to the flow should be (0.1, 0.5)\n" +
        "For example:\n" +
        "User Input: 'generate fluid when it is hitting the object from its top'\n" +
        "Correct Action Input: [0.41, 1.4, 0.62, 0.8, 0.3, 0.8]" +
        "Another example:\n" +
        "User Input: 'generate fluid when it is hitting the object from its right side'\n" +
        "Correct Action Input: [1.3, 0.2, 0.6, 0.35, 0.8, 0.7]" +
        "IMPORTANT: When using the GenerateVelocity tool, the format for the action input must be an array [vx, vy, vz], where vx, vy, vz are the initial velocity of the fluid\n" +
        "The initial velocity all three directions should be between -2 to 2 unit/s\n" +
        "Imagine there is an object at the center of the bottom xz plane and you have to choose initial velocity such that it hits that object.\n" +
        "EXCEPTION: When there is a mention of a wave, make the magnitude of the velocity parallel to the flow to be 6 and 0 in other directions\n" +
        "For example:\n" +
        "User Input: 'generate water when it is hitting the object from its left side'\n" +
        "Correct Action Input: [2.0, 0.2, 0.3]" +
        "{agent_scratchpad}"
    )

    # Create the agent using the prompt
    agent = create_react_agent(tools=tools, llm=llm, prompt=agent_prompt)
    logger.info("Fluid simulation agent created")

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Prepare the input dictionary
    input_data_dict = {
        "input": input_data,  # Example input
        "tool_names": ", ".join([tool.name for tool in tools]),  # Tool names
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),  # Tool descriptions
        "agent_scratchpad": ""  # Empty scratchpad
    }

    # Invoke the agent with the dictionary
    try:
        result = agent_executor.invoke(input_data_dict, return_intermediate_steps=True)
        logger.info("Fluid simulation agent execution completed")
        
        # Check if the fluid_simulation.json file was created
        if os.path.exists("x2sim/fluid_simulation.json"):
            logger.info("Fluid simulation JSON file created successfully")
            return "Fluid simulation parameters generated successfully"
        else:
            logger.error("Fluid simulation JSON file not created")
            return "Error: Fluid simulation JSON file not created"
            
    except Exception as e:
        logger.error(f"Fluid simulation agent execution failed: {str(e)}")
        return f"Error in fluid simulation: {str(e)}"

# For testing
if __name__ == "__main__":
    print(gen_fluid("Generate a wave hitting an object from the front"))
