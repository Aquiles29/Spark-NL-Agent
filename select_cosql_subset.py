import json

SIGNATURES = [
    "NONE",
    "JOIN",
    "GROUP_BY",
    "JOIN + GROUP_BY",
    "GROUP_BY + HAVING",
    "JOIN + GROUP_BY + HAVING",
]

with open(
    "cosql_transfer_results.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

records = data["records"]

selected = []

for signature in SIGNATURES:

    candidates = [
        r for r in records
        if r["gold_signature"] == signature
    ]

    candidates = sorted(
        candidates,
        key=lambda r: r["dialogue_index"]
    )

    if not candidates:
        print(f"WARNING: no examples for {signature}")
        continue

    # Same deterministic principle as BIRD:
    # median identifier, independent of model outcome.
    chosen = candidates[len(candidates) // 2]

    selected.append(chosen)

    print("=" * 80)
    print(f"SIGNATURE: {signature}")
    print(f"Candidates: {len(candidates)}")
    print(f"Dialogue index: {chosen['dialogue_index']}")
    print(f"Database: {chosen['db_id']}")
    print(f"Question: {chosen['question']}")
    print(f"Gold: {chosen['gold_signature']}")
    print(f"Predicted: {chosen['predicted_signature']}")
    print()

with open(
    "cosql_evaluation_sample.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        selected,
        f,
        indent=4,
        ensure_ascii=False
    )

print("=" * 80)
print(f"Selected {len(selected)} CoSQL queries.")
print("Saved to cosql_evaluation_sample.json")