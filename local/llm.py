from local.keys import silicon_key
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import json
import re
import ast

def get_llm_answer(text: str, model_name: str):
    client = OpenAI(api_key=silicon_key, 
                    base_url="https://api.siliconflow.cn/v1"
                    )
    message = {
        'role': 'user',
        'content': text
    }
    response = client.chat.completions.create(
        model = model_name,
        messages = [message],
        stream = False
    )
    
    content = response.choices[0].message.content
    return content

description_prompts = "Please describe the main characteristics of the image patch from satellite in a sentence. Its semantic label is: "

def get_vlm_description(img: Image, label: str, model_name: str):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    prompts = description_prompts + label + ". Only describe observable content. A cautious guess of building type can be included if clearly supported by visual evidence. Mention objects, their attributes (size, shape, color), and spatial relations (e.g., next to, surrounded by, aligned with). Start the description with a noun phrase (e.g., 'A building...')."
    
    client = OpenAI(api_key=silicon_key, 
                    base_url="https://api.siliconflow.cn/v1"
                    )
    message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": prompts
            }
        ]
    }
    response = client.chat.completions.create(
        model=model_name,
        messages = [message],
        stream=False
    )
    
    content = response.choices[0].message.content
    return content

task_parsing_prompts = """Please parse the photogrammetry task description in natural language. The photogrammetry survey task demands and task regions are needed. Output the results in json format:
{
    "tasks": [
        {
            "task_demands": "string",
            "task_region": "string"
        }
    ]
}
Requirements:
 - Each task must be a separate object in the list.
 -"task_demands" should include key requirements (e.g., resolution, overlap).
 -"task_region" should describe the task area in natural language.
 - There can be multiple tasks or just a single task.
 - Only output valid JSON.
The natural-language-task you need to parse is: """

def task_parsing(task_text: str, model_name: str):
    prompts = task_parsing_prompts + task_text
    answer = get_llm_answer(prompts, model_name)
    answer = re.sub(r"```json|```", "", answer).strip()
    
    data = json.loads(answer)
    task_demands = [task["task_demands"] for task in data["tasks"]]
    task_regions = [task["task_region"] for task in data["tasks"]]
    return task_demands, task_regions

def params_reasoning(task_demands, task_regions, model_name: str):
    input_data = {
        "task_demands": task_demands,
        "task_regions": task_regions
    }
    
    prompts = f"""Given the following photogrammetry tasks:
    {json.dumps(input_data, indent=2)}""" + """
    Please infer the required parameters for each task (max_interval(m), expand_rate, flight_interval(m), flight_speed(m/s)), and answer them in corresponding task lists.
    Parameter definitions:
     - max_interval (meters): maximum width between adjacent flight strips (flight lines).
     - expand_rate (>1): expansion ratio applied to the survey region boundary to ensure full coverage.
     - flight_interval (meters): maximum spacing between adjacent image capture points along the flight path, ensuring sufficient overlap.
     - flight_speed (m/s): UAV flight speed during photogrammetry.
     
    Guidelines:
    - High-resolution tasks require smaller flight_interval and smaller max_interval.
    
    Requirements:
    - Output parameters for each task in order.
    - The number of parameter sets must match the number of tasks.
    - Do not include explanation.
    - Output only a list of list(s), corresponding to the task(s).
    - The output must start with '[' and end with ']'.
    
    Output format:
    [[max_interval, expand_rate, flight_interval, flight_speed],[max_interval, expand_rate, flight_interval, flight_speed],...]]
    
    Output format example:
    [[5.0, 1.1, 3.0, 4.0], [8.0, 1.05, 5.0, 6.0]]
    """
    
    answer = get_llm_answer(prompts, model_name)
    match = re.search(r"\[\[.*\]\]", answer, re.S)
    if match:
        params = ast.literal_eval(match.group())
    
    return params # params

def survey_areas_reasoning(task_demands, task_regions, scene_graph: str, model_name: str):
    input_data = {
        "task_demands": task_demands,
        "task_regions": task_regions
    }
    prompts = f"""Given the following photogrammetry tasks:
    {json.dumps(input_data, indent=2)}
    And the following scene graph (in text form):
    {scene_graph}
    """ + """
    Task:
    For each task_region, find the corresponding node id(s) as survey area(s) in the scene graph.
    Requirements:
    - Output a list of lists.
    - Each inner list contains the node id(s) corresponding to one task.
    - The order must match the order of tasks.
    - Each task can have one or multiple node ids.
    - If no exact match, return the closest relevant node.
    - Do not include explanation.
    - Output must start with '[' and end with ']'.
    
    Output format example:
    [[3], [5, 7], [2]]
    """
    
    answer = get_llm_answer(prompts, model_name)
    match = re.search(r"\[\[.*\]\]", answer, re.S)
    if match:
        survey_areas = ast.literal_eval(match.group())
    return survey_areas # survey_areas

def routes_reasoning(params,
                     survey_areas,
                     start_point, # (lon, lat)
                     scene_graph: str,
                     model_name: str
                     ):
    input_data = {
        "task_params": params,
        "survey_areas": survey_areas,
        "start_point": start_point
    }
    prompts = f"""Given the following tasks and parameters:
    {json.dumps(input_data, indent=2)}
    And the scene graph:
    {scene_graph}
    Task:
    Plan an efficient execution order of tasks and routing paths.
    
    Definitions:
    - Each survey_area is a list of node ids representing the task region.
    - You may reorder tasks to minimize total travel distance.
    - The start point is given as coordinates; assume the nearest node in the graph is the starting node.
    
    Routing rules:
    - Each route represents the path between two consecutive tasks (exception: the starting and first task).
    - Each route is a list of node ids in visiting order.
    - The first route starts from the node closest to the start_point.
    - The last node of each route must be a nearest node in the next task's survey_area, only one node of that area (because the area may include several nodes).
    - The first node of each route must be a node in the previous task's survey_area, only one node of that area representing the departure of that area. For the first route, the first node is the start_point.
    - After finishing a task, move to the next task via graph connections.
    - The final task does not need an outgoing route.
    
    Requirements:
    - Optimize for shortest total path (greedy or approximate is acceptable).
    - Reorder tasks if needed.
    - Keep task_params, survey_areas and routes aligned after reordering. Each route's last node represents the corresponding survey_area.
    
    Output format (strict JSON):
    {{
    "task_params": [[...], [...]],
    "survey_areas": [[...], [...]],
    "routes": [[node_id, ...], [node_id, ...]]
    }}
    Do not include explanation.
    Only output valid JSON.
    """
    
    answer = get_llm_answer(prompts, model_name)
    answer = re.sub(r"```json|```", "", answer).strip()
    data = json.loads(answer)
    task_params = data["task_params"]
    survey_areas = data["survey_areas"]
    routes = data["routes"]
    
    return task_params, survey_areas, routes