import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "knowledge_graph"
    / "data"
    / "nodes"
    / "all_nodes.json"
)


def main():

    print("=" * 70)
    print("INSPECTING KNOWLEDGE GRAPH SCHEMA")
    print("=" * 70)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print()
    print("TOP-LEVEL KEYS")
    print("-" * 70)

    for key in data.keys():
        value = data[key]

        if isinstance(value, list):
            print(f"{key:<20} : list ({len(value)} items)")
        elif isinstance(value, dict):
            print(f"{key:<20} : dict")
        else:
            print(f"{key:<20} : {type(value).__name__}")

    # ---------------------------------------------------------
    # Inspect nodes
    # ---------------------------------------------------------

    nodes = data.get("nodes", [])

    print()
    print("FIRST 5 NODE RECORDS")
    print("-" * 70)

    for i, node in enumerate(nodes[:5], start=1):

        print()
        print(f"NODE {i}")
        print(json.dumps(
            node,
            indent=2,
            ensure_ascii=False
        ))

    # ---------------------------------------------------------
    # Inspect relationships
    # ---------------------------------------------------------

    relationships = data.get(
        "relationships",
        []
    )

    print()
    print("FIRST 5 RELATIONSHIP RECORDS")
    print("-" * 70)

    for i, relationship in enumerate(
        relationships[:5],
        start=1
    ):

        print()
        print(f"RELATIONSHIP {i}")
        print(json.dumps(
            relationship,
            indent=2,
            ensure_ascii=False
        ))

    # ---------------------------------------------------------
    # Find keys across nodes
    # ---------------------------------------------------------

    node_keys = set()

    for node in nodes:
        node_keys.update(node.keys())

    print()
    print("ALL NODE KEYS")
    print("-" * 70)

    for key in sorted(node_keys):
        print(key)

    # ---------------------------------------------------------
    # Find keys across relationships
    # ---------------------------------------------------------

    relationship_keys = set()

    for relationship in relationships:
        relationship_keys.update(
            relationship.keys()
        )

    print()
    print("ALL RELATIONSHIP KEYS")
    print("-" * 70)

    for key in sorted(relationship_keys):
        print(key)

    print()
    print("=" * 70)
    print("SCHEMA INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()