# ----- CONFIG -----
import json
import time
import numpy as np
from pydantic import BaseModel, Field
from ollama import chat, embeddings
from pathlib import Path

# ----- FILE PATHS -----
script_folder = Path(__file__).parent.resolve()
tools_list_path = script_folder / 'input' / 'tools_list.json'
queries_list_path = script_folder / 'input' / 'queries_list.json' 
results_path = script_folder / 'output' / 'results.json'

# ----- VARIABLES -----
model = "llama3:latest"
ratio_queries = 0.1 #50%
embedding_model = "mxbai-embed-large"

# ----- PYDANTIC SCHEMA -----
class RouteDecision(BaseModel):
    selected_tool: str = Field(description="The exact name of the tool. Return 'none' if no tool matches.")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0")
    reasoning: str = Field(description="A short sentence explaining why this tool was chosen")

# ----- RAG LOGIC -----
def get_vector(text: str) -> list: 
    return embeddings(model=embedding_model, prompt=text)['embedding']

def cosine_similarity(a, b):
    return np.dot(a ,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrive_top_tools(user_prompt:str, tools_list: list, top_k: int = 3) -> list:

    query_vec = get_vector(user_prompt)
    scrored_tools = []

    for tool in tools_list:
        tool_desc = tool.get('description', str(tool))
        tool_vec = get_vector(tool_desc)
        score = cosine_similarity(query_vec, tool_vec)
        scrored_tools.append((score, tool))

    scrored_tools.sort(key=lambda x: x[0], reverse=True)
    top_result = scrored_tools[:top_k]

    # ----- LOG RAG -----
    for i,(score,tool) in enumerate(top_result):
        tool_name = tool.get('name','unknow_name')
        print(f"{i+1} Score: {score:.4f} tools: {tool_name}")


    return [t[1] for t in top_result]

# ----- AGENT LOGIC -----
def agent_router(user_prompt: str, manifest: dict, model_name: str) -> RouteDecision:
    
    relevant_tools = retrive_top_tools(user_prompt, manifest, top_k=3)


    system_prompt = f"""You are a Router Agent expert in medical and dental imaging (CBCT, IOS, MRI).
    Your role is to analyze the user's request and select the most relevant tool from the FILTERED list below.
    If none of these {len(relevant_tools)} tools fit, return 'none'.

    Relevant Tools Subset:{json.dumps(relevant_tools, indent=2)}
    """
    
    response = chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        format=RouteDecision.model_json_schema(),
        options={"temperature": 0},
    )
    
    return RouteDecision.model_validate_json(response.message.content)

# ----- MAIN -----
def main():

    # ----- LOAD FILES -----
    with open(tools_list_path, 'r', encoding='utf-8') as f:
        tools_list = json.load(f)

    with open(queries_list_path, 'r', encoding='utf-8') as f:
        queries_list = json.load(f)

    # ----- TRACKING VARIABLES -----
    results_detail = []
    correct_count = 0
    total_time = 0.0

    # ----- MAIN LOOP -----
    limit = int(len(queries_list) * ratio_queries)
    queries_to_run = queries_list[:limit]

    for prompt, expected_tool in queries_to_run:
        print(f"\nPrompt: {prompt}")

        # ----- RUN ROUTER -----
        t0 = time.time()
        decision = agent_router(prompt, tools_list, model)
        t1 = time.time()

        # ----- LOCAL RESULTS -----
        latency = t1 - t0
        total_time += latency
        is_correct = decision.selected_tool == expected_tool
        
        if is_correct:
            correct_count += 1

        status_icon = "✅" if decision.selected_tool == expected_tool else "❌"
        print(f"{status_icon} Tool chosen: {decision.selected_tool} (Expected: {expected_tool})")
        print(f"Confidence : {decision.confidence * 100:.2f}%")
        print(f"Time       : {t1 - t0:.2f} seconds")
        print(f"Reason : {decision.reasoning}")

        results_detail.append({
            "prompt": prompt,
            "expected_tool": expected_tool,
            "Selected_tool": decision.selected_tool,
            "Correct?": is_correct,
            "Confidence": decision.confidence,
            "Latency": round(latency, 4),
            "reasoning": decision.reasoning
        })

    # ----- GLOBAL RESULTS (json) -----
    total_queries = len(queries_to_run)
    accuracy = (correct_count / total_queries) * 100 if total_queries > 0 else 0.0
    avg_time = (total_time / total_queries) if total_queries > 0 else 0.0

    summary = {
        "model": model,
        "metrics":{
            "total_queries": total_queries,
            "correct_predictions": correct_count,
            "accuracy_percentage": round(accuracy,2),
            "average_latency_seconds": round(avg_time, 4),
            "total_execution_time_seconds": round(total_time, 4),
        },
        "details": results_detail,
    }

    # ----- SAVE RESULTS FILE -----
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n BENCHMARK DONE")
        print(f"Accuracy: {accuracy:.2f}% ({correct_count/total_queries})")
        print(f"Average Time: {avg_time:.2f} secondes")
        print(f"Save in: {results_path.name}")

if __name__ == "__main__":
    main()
