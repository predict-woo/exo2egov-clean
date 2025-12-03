#!/usr/bin/env python3
"""
Convert single-exo JSONL to multi-exo JSONL.
Replaces the single cam2 exo view with a list of 4 exo views (cam0, cam1, cam2, cam3).
"""

import json
import argparse
from pathlib import Path


def convert_to_multi_exo(input_path: str, output_path: str = None):
    """
    Read a JSONL file with single exo view and create one with 4 exo views.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (default: input_multi_exo.jsonl)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_multi_exo.jsonl"
    else:
        output_path = Path(output_path)
    
    exo_cams = ["cam0", "cam1", "cam2", "cam3"]
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            data = json.loads(line)
            exo_path = data["exo"]
            ego_path = data["ego"]
            
            # Generate 4 exo views by replacing cam2 with cam0, cam1, cam2, cam3
            exo_views = []
            for cam in exo_cams:
                # Replace cam2 (or any camX) with the target camera
                exo_view = exo_path.replace("/cam2/", f"/{cam}/")
                exo_views.append(exo_view)
            
            # Create new entry with list of exo views
            new_data = {
                "exo": exo_views,
                "ego": ego_path
            }
            
            f_out.write(json.dumps(new_data) + "\n")
    
    print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert single-exo JSONL to multi-exo JSONL")
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("-o", "--output", help="Output JSONL file path (optional)")
    
    args = parser.parse_args()
    convert_to_multi_exo(args.input, args.output)

