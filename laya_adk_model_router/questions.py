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
                "Greetings, small talk, routine FAQs, basic factual lookups, "
                "elementary science or general-knowledge multiple-choice recall, "
                "definitions, basic status checks, short translations, reformatting, "
                "or direct single-step questions that need no multi-step calculation."
            ),
            "complex": (
                "Multi-step math word problems, arithmetic or algebraic calculations, "
                "quantitative reasoning, Python/SQL code generation or refactoring, "
                "debugging, stack traces, kernel panics, log parsing, race conditions, "
                "memory leaks, system architecture, root-cause analysis, performance "
                "optimization, math/logic proofs, or multi-step analytical reasoning."
            ),
        },
    }
}
