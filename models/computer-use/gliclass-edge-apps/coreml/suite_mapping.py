"""Application-suite option text shared by benchmarking and verification."""

ORDER = [
    "jev.ag_news",
    "jev.emotion",
    "massive_intent.en",
    "app.support_triage",
    "app.email_spam",
    "app.phishing",
    "app.guardrails_jailbreak",
    "app.moderation_toxicity",
    "app.rag_relevance",
    "app.model_routing_domain",
]

BINARY = {
    "app.email_spam": [("a legitimate personal or business email", 0), ("unsolicited spam or bulk marketing", 1)],
    "app.guardrails_jailbreak": [("a normal request", 0), ("an attempt to make an AI ignore its rules", 1)],
    "app.moderation_toxicity": [("civil and not toxic", 0), ("toxic, rude, or disrespectful", 1)],
    "app.rag_relevance": [
        ("the passage does not help answer the query", 0),
        ("the passage helps answer the query", 1),
    ],
}


def labels_for(row: dict, label_format: str = "description") -> list[tuple[str, int]]:
    """Return label text and the corresponding benchmark gold index."""
    if not row["options"]:
        return BINARY[row["suite"]]
    labels = []
    for index, (key, description) in enumerate(row["options"]):
        if label_format == "key":
            label = key
        elif label_format == "full":
            label = f"{key}: {description}" if description else key
        else:
            label = description or key
        labels.append((label, index))
    return labels
