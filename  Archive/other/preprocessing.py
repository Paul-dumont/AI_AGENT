import pandas as pd
import yaml
import json
import re
from difflib import SequenceMatcher

queries = pd.read_excel("Excel_queries/queries.xlsx","Preprocess ready")
manifest_file = "manifest/manifest.yaml"
with open(manifest_file, "r") as f:
        manifest = yaml.safe_load(f)

columns_to_ignore = ["Horodateur","Your name","Your level of expertise with these medical tools","Column 27"]

scripts = manifest.get("scripts", [])

dic_yaml = {}

for tool in scripts:
    name = tool.get("name","")
    description = tool.get("description","")
    tags = tool.get("tags",[])

    dic_yaml[name] = {"Description": description,"Tags":tags}

def normalize_name(name):
    """Convert Excel column name to YAML tool name format"""
    normalized = name.lower().replace(" ", "_")
    normalized = re.sub(r'[^\w_]', '', normalized)
    return normalized

def fuzzy_match(excel_col, yaml_tools, threshold=0.7):
    """Find best matching YAML tool for an Excel column using fuzzy matching"""
    normalized_excel = normalize_name(excel_col)
    
    # Exact match first
    for tool_name in yaml_tools.keys():
        if normalize_name(tool_name) == normalized_excel:
            return tool_name
    
    # Fuzzy match if exact match fails
    best_match = None
    best_score = threshold
    
    for tool_name in yaml_tools.keys():
        normalized_tool = normalize_name(tool_name)
        # Compare both original and normalized versions
        score1 = SequenceMatcher(None, normalized_excel, normalized_tool).ratio()
        score2 = SequenceMatcher(None, excel_col.lower(), tool_name.lower()).ratio()
        score = max(score1, score2)
        
        if score > best_score:
            best_score = score
            best_match = tool_name
    
    return best_match

column_to_tool = {}
unmatched_cols = []

for col in queries.columns:
    if col not in columns_to_ignore:
        matched_tool = fuzzy_match(col, dic_yaml, threshold=0.6)
        if matched_tool:
            column_to_tool[col] = matched_tool
        else:
            unmatched_cols.append(col)

print(f"📍 Column to Tool Mapping:")
print(f"Found {len(column_to_tool)} matches out of {len(queries.columns) - len(columns_to_ignore)}")
if unmatched_cols:
    print(f"❌ Unmatched columns: {unmatched_cols}")

# Build simplified YAML manifest for system prompt
simplified_manifest = {
    "tools": {name: info for name, info in dic_yaml.items()}
}

# Build the system prompt with YAML manifest and expected JSON format
system_base = """You are a CLI Router expert. You need to choose the most appropriate tool answering to the user query.

# Available Tools:
"""

for tool_name, tool_info in dic_yaml.items():
    system_base += f"  - name: {tool_name}\n"
    system_base += f"    description: {tool_info.get('Description', 'N/A')}\n"
    tags = tool_info.get('Tags', [])
    if tags:
        system_base += f"    tags: [{', '.join(tags)}]\n"

system_base += """```

# Expected Output Format:
You MUST respond with a valid JSON object in this exact format:
```json
{
  "tool_name": "<selected_tool_name>"
}
```

Only provide the JSON output, no additional text."""

training_dataset = []

# Create training samples
for idx, row in queries.iterrows():
    for col, tool_name in column_to_tool.items():

        user_query = row[col]

        if pd.isna(user_query) or user_query == "":
            continue

        sample = {
            "system": system_base,
            "user": str(user_query),
            "assistant": {
                "tool_name": tool_name
            }
        }
        training_dataset.append(sample)

# Save to JSON file
output_file = "training_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(training_dataset, f, indent=2, ensure_ascii=False)