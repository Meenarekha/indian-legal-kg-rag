import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

KG_PATH = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "knowledge_graph.json"
)


def main():

    print("=" * 70)
    print("KNOWLEDGE GRAPH SECTION INSPECTION")
    print("=" * 70)

    with open(KG_PATH, "r", encoding="utf-8") as file:
        graph = json.load(file)

    nodes = graph.get("nodes", [])

    sections = [
        node
        for node in nodes
        if node.get("node_type") == "Section"
    ]

    print(f"\nTotal Section nodes: {len(sections)}")

    print("\nFirst 30 Section nodes:")
    print("-" * 70)

    for i, node in enumerate(sections[:30], start=1):

        print(f"\n{i}. Node ID: {node.get('node_id')}")

        print(
            f"   Properties: "
            f"{node.get('properties', {})}"
        )

    print("\n" + "=" * 70)

    # Specifically search for Section 438
    matches = []

    for node in sections:

        text = (
            str(node.get("node_id", "")) +
            " " +
            str(node.get("properties", {}))
        ).lower()

        if "438" in text:

            matches.append(node)

    print(
        f"\nNodes containing '438': "
        f"{len(matches)}"
    )

    for node in matches:

        print("\nMATCH:")
        print(f"Node ID    : {node.get('node_id')}")
        print(f"Properties : {node.get('properties')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()