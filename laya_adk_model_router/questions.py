from typing import Any

# Define the Laya System-1 Classification Questions.
# Update the instructions and criteria below to match your specific domain or use case.
LAYA_COMPLEXITY_QUESTIONS: dict[str, dict[str, Any]] = {
    "complexity": {
        "type": "choice",
        "instructions": (
            "How much reasoning depth is required to answer `user_message` "
            "(taking `recent_context` into account if present)?"
        ),
        "criteria": {
            "simple": (
                "Greetings, small talk, routine FAQs, factual lookups, definitions, "
                "basic status checks, short translations, reformatting, or direct "
                "single-step questions that need no deep reasoning."
            ),
            "complex": (
                "Debugging, stack traces, kernel panics, log parsing, race conditions, "
                "memory leaks, code generation or refactoring, system architecture, "
                "root-cause analysis, performance optimization, math/logic proofs, "
                "SQL/data analysis, or multi-step analytical reasoning."
            ),
        },
    }
}
