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
                "Greetings, routine FAQs, elementary 4-option multiple-choice recall, "
                "brief everyday arithmetic word problems (counting items, basic shopping/sales totals), "
                "or simple one-liner Python helper snippets (`Provide a brief`)."
            ),
            "complex": (
                "Complex multi-paragraph analytical scenarios with 8 to 10 options (A through J), "
                "competition or Olympiad-level mathematics problems with LaTeX derivations (`\\frac`, `$...$`), "
                "or full algorithmic Python engineering specifications with multi-example docstrings and time/space complexity analysis."
            ),
        },
    }
}

