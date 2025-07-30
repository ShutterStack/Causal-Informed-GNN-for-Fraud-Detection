import json
import nbformat
import argparse

def convert_json_to_ipynb(input_json_path: str, output_ipynb_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        notebook_json = json.load(f)

    nb = nbformat.from_dict(notebook_json)

    with open(output_ipynb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Notebook saved to: {output_ipynb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a JSON notebook into .ipynb format")
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument("output_ipynb", help="Path to output .ipynb file")
    args = parser.parse_args()

    convert_json_to_ipynb(args.input_json, args.output_ipynb)
