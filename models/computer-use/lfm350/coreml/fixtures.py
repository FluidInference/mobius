"""One real-text smoke request for parity and conversion."""

SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["billing", "technical", "sales"]},
        "urgent": {"type": "boolean"},
    },
    "required": ["route", "urgent"],
    "additionalProperties": False,
}
CONTEXT = "Our company invoice contains a duplicate charge, and payment is due tomorrow."
