#!/usr/bin/env python3
"""
Convertit un fichier JSON en JSONL formaté pour l'entraînement LLM.
Support de plusieurs formats de sortie.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def load_json(filepath: str) -> List[Dict]:
    """Charge un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_conversation(item: Dict) -> Dict:
    """
    Format de conversation (style OpenAI chat).
    Chaque exemple devient une conversation avec messages.
    """
    return {
        "messages": [
            {"role": "system", "content": item.get("system", "")},
            {"role": "user", "content": item.get("user", "")},
            {"role": "assistant", "content": json.dumps(item.get("assistant", {}))}
        ]
    }


def format_instruction_output(item: Dict) -> Dict:
    """
    Format instruction/output.
    Combine system et user comme instruction, assistant comme output.
    """
    instruction = f"{item.get('system', '')}\n\nUser Query: {item.get('user', '')}"
    output = json.dumps(item.get("assistant", {}))
    
    return {
        "instruction": instruction.strip(),
        "output": output
    }


def format_prompt_completion(item: Dict) -> Dict:
    """
    Format prompt/completion.
    Minimal, destiné aux modèles GPT-2/3 style.
    """
    prompt = f"[SYSTEM]\n{item.get('system', '')}\n\n[USER]\n{item.get('user', '')}\n\n[ASSISTANT]\n"
    completion = json.dumps(item.get("assistant", {}))
    
    return {
        "prompt": prompt.strip(),
        "completion": completion
    }


def format_compact(item: Dict) -> Dict:
    """
    Format compact.
    Idéal pour les modèles de classification fine-tuning.
    """
    return {
        "input": item.get("user", ""),
        "output": item.get("assistant", {}),
        "system": item.get("system", "")[:500]  # Limiter système pour compacité
    }


def save_jsonl(data: List[Dict], output_filepath: str) -> None:
    """Sauvegarde une liste de dictionnaires en JSONL."""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved: {output_filepath} ({len(data)} lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Convertit JSON en JSONL pour l'entraînement LLM"
    )
    parser.add_argument(
        "input_file",
        help="Fichier JSON d'entrée"
    )
    parser.add_argument(
        "-o", "--output",
        help="Fichier JSONL de sortie (défaut: input_file.jsonl)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["conversation", "instruction", "prompt", "compact"],
        default="conversation",
        help="Format de sortie (défaut: conversation)"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Générer tous les formats de sortie"
    )
    
    args = parser.parse_args()
    
    # Charger les données
    print(f"Loading: {args.input_file}")
    data = load_json(args.input_file)
    print(f"Loaded: {len(data)} examples")
    
    # Déterminer le chemin de sortie
    input_path = Path(args.input_file)
    output_dir = input_path.parent
    base_name = input_path.stem
    
    # Formatters disponibles
    formatters = {
        "conversation": format_conversation,
        "instruction": format_instruction_output,
        "prompt": format_prompt_completion,
        "compact": format_compact
    }
    
    if args.all:
        # Générer tous les formats
        for format_name, formatter in formatters.items():
            formatted_data = [formatter(item) for item in data]
            output_file = output_dir / f"{base_name}_{format_name}.jsonl"
            save_jsonl(formatted_data, str(output_file))
    else:
        # Générer le format spécifié
        formatter = formatters[args.format]
        formatted_data = [formatter(item) for item in data]
        
        if args.output:
            output_file = args.output
        else:
            output_file = output_dir / f"{base_name}_{args.format}.jsonl"
        
        save_jsonl(formatted_data, str(output_file))
    
    print("\n📊 Conversion terminée!")


if __name__ == "__main__":
    main()
