# ----- CONFIG -----
import json
import time
from pydantic import BaseModel, Field
from ollama import chat
from pathlib import Path

# ----- VARIABLES -----
model = "llama3:latest" #
ratio_queries = 1 #%
queries_type = "tool_only" 
# queries_type = "tool_param_expert_vocab" 


# ----- FILE PATHS -----
script_folder = Path(__file__).parent.resolve()
tools_list_path = script_folder.parent / 'input' / 'tools_list.json'
queries_list_path = script_folder.parent / 'input' / 'user_queries' / f'{queries_type}.json'
results_path = script_folder / 'output' / f'{model.replace(":", "-")}_{queries_type}.json'

# ----- PYDANTIC SCHEMA -----
class RouteDecision(BaseModel):
    selected_tool: str = Field(description="The exact name of the tool. Return 'none' if no tool matches.")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0")
    reasoning: str = Field(description="A short sentence explaining why this tool was chosen")

# ----- AGENT LOGIC -----
def agent_router(user_prompt: str, manifest: dict, model_name: str) -> RouteDecision:
    
    system_prompt = f"""You are a Router Agent expert in medical and dental imaging (CBCT, IOS, MRI).
    Your role is to analyze the user's request and select the most relevant tool from the provided list.

    Here is the complete tool manifest:{json.dumps(manifest, indent=2)}
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
