import pandas as pd

from src.evaluation import execution_accuracy


gold = pd.DataFrame({
    "rate": [
        0.0434782608695652,
        0.0704225352112676,
        0.113636363636364
    ]
})

predicted = [
    ("0.0434782608695652",),
    ("0.0704225352112676",),
    ("0.113636363636364",)
]


ea = execution_accuracy(
    gold,
    predicted
)

print("Execution Accuracy:", ea)

wrong = [
    ("0.0434782608695652",),
    ("0.0704225352112676",),
    ("0.5000000000000000",)
]

print(
    "Wrong result:",
    execution_accuracy(gold, wrong)
)