import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "relationships"
    / "internal_citation_text_report.json"
)


def main():

    print("=" * 60)
    print("INSPECTING INTERNAL CITATION REPORT")
    print("=" * 60)

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print()
    print("TOP-LEVEL TYPE")
    print("-" * 60)
    print(type(data).__name__)

    if isinstance(data, dict):

        print()
        print("TOP-LEVEL KEYS")
        print("-" * 60)

        for key in data.keys():
            print(f"- {key}")

        print()
        print("STRUCTURE")
        print("-" * 60)

        for key, value in data.items():

            print(f"\n[{key}]")
            print(f"Type: {type(value).__name__}")

            if isinstance(value, list):
                print(f"Length: {len(value)}")

                if len(value) > 0:
                    print("First item:")
                    print(value[0])

            elif isinstance(value, dict):
                print("Keys:")
                for subkey in value.keys():
                    print(f"  - {subkey}")

            else:
                print(f"Value: {value}")

    elif isinstance(data, list):

        print()
        print(f"List length: {len(data)}")

        if len(data) > 0:
            print()
            print("FIRST ITEM")
            print("-" * 60)
            print(data[0])

    print()
    print("=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()