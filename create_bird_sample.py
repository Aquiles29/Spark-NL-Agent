import json
import random
import pandas as pd


INPUT_FILE = "bird_clause_categories.json"
OUTPUT_FILE = "bird_evaluation_sample.json"

AVAILABLE_DATABASES = {
    "california_schools",
    "student_club",
    "superhero",
    "thrombosis_prediction",
    "toxicology",
}

RANDOM_SEED = 42


# Number of examples to select from each structural category.
# We can adjust these later depending on API availability.
SAMPLE_SIZES = {
    "NONE": 5,
    "JOIN": 5,
    "GROUP_BY": 5,
    "JOIN+GROUP_BY": 5,
    "GROUP_BY+HAVING": 1,
    "JOIN+GROUP_BY+HAVING": 5,
}


def main():

    random.seed(RANDOM_SEED)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [
        item
        for item in data
        if item["db_id"] in AVAILABLE_DATABASES
    ]
    
    df = pd.DataFrame(data)

    selected = []

    print("\n" + "=" * 60)
    print("CREATING STRATIFIED BIRD SAMPLE")
    print("=" * 60)

    for category, requested_n in SAMPLE_SIZES.items():

        category_df = df[
            df["category_signature"] == category
        ]

        available = len(category_df)

        n = min(requested_n, available)

        sample = category_df.sample(
            n=n,
            random_state=RANDOM_SEED
        )

        selected.extend(sample.to_dict(orient="records"))

        print(
            f"{category:30s}: "
            f"selected {n:2d} / {available:4d} available"
        )

    # Sort only to make the file easier to inspect
    selected = sorted(
        selected,
        key=lambda x: (
            x["category_signature"],
            x["question_id"]
        )
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            selected,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "-" * 60)
    print(f"Total selected: {len(selected)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()