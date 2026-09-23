"""Small public request for native and Core ML parity checks."""


def fixture() -> dict:
    return {
        "states": [
            {
                "id": "example",
                "state": "The target is left of the aim.",
                "questions": {
                    "action": {
                        "type": "choice",
                        "instructions": "Choose the next action.",
                        "criteria": {
                            "left": "Move the aim left.",
                            "right": "Move the aim right.",
                            "shoot": "Fire.",
                            "noop": "Wait.",
                        },
                    }
                },
            }
        ]
    }


def requests() -> list[dict]:
    boolean = {
        "states": [
            {
                "id": "boolean",
                "state": "The invoice says paid and shows a zero balance.",
                "questions": {
                    "paid": {"type": "boolean", "instructions": "Has the invoice been paid?"}
                },
            }
        ]
    }
    score = {
        "states": [
            {
                "id": "score",
                "state": "The order arrived two weeks late and the box was damaged.",
                "questions": {
                    "severity": {
                        "type": "score",
                        "instructions": "Rate customer impact.",
                        "criteria": ["Low impact", "Moderate impact", "High impact"],
                    }
                },
            }
        ]
    }
    return [fixture(), boolean, score]
