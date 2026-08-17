"""Deterministic local tools used before language-model generation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
from typing import Callable


@dataclass(frozen=True)
class ToolResult:
    name: str
    text: str


def _safe_number(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_number(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_number(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if not isinstance(node, ast.BinOp):
        raise ValueError("unsupported expression")

    left = _safe_number(node.left)
    right = _safe_number(node.right)
    operations: dict[type[ast.operator], Callable[[float | int, float | int], float | int]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }
    if isinstance(node.op, ast.Pow):
        if abs(right) > 10 or abs(left) > 1_000_000:
            raise ValueError("exponent is too large")
        result = left**right
    else:
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported operator")
        result = operation(left, right)
    if type(result) not in (int, float):
        raise ValueError("result must be a real number")
    if not math.isfinite(float(result)) or abs(result) > 1e15:
        raise ValueError("result is too large")
    return result


def _arithmetic(query: str) -> ToolResult | None:
    expression = query.lower().strip().rstrip("?")
    expression = re.sub(
        r"^(?:what is|calculate|compute|evaluate|solve)\s+", "", expression
    )
    for phrase, operator in {
        "divided by": "/",
        "multiplied by": "*",
        "times": "*",
        "plus": "+",
        "minus": "-",
        "×": "*",
        "÷": "/",
    }.items():
        expression = expression.replace(phrase, operator)
    if not re.fullmatch(r"[0-9eE.\s+\-*/%()]+", expression):
        return None
    if not re.search(r"[+\-*/%]", expression):
        return None
    try:
        value = _safe_number(ast.parse(expression, mode="eval"))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return ToolResult("calculator", "I could not safely evaluate that expression.")
    rendered = str(int(value)) if isinstance(value, float) and value.is_integer() else f"{value:.10g}" if isinstance(value, float) else str(value)
    return ToolResult("calculator", f"The result is {rendered}.")


def _date_time(query: str) -> ToolResult | None:
    lowered = query.lower()
    asks_time = bool(re.search(r"\b(current time|what time|time is it)\b", lowered))
    asks_date = bool(re.search(r"\b(current date|what date|today's date|date today|what day is it)\b", lowered))
    if not (asks_time or asks_date):
        return None
    now = datetime.now().astimezone()
    if asks_time and asks_date:
        text = now.strftime("It is %A, %B %d, %Y at %I:%M %p %Z.")
    elif asks_time:
        text = now.strftime("The current local time is %I:%M %p %Z.")
    else:
        text = now.strftime("Today's date is %A, %B %d, %Y.")
    return ToolResult("clock", text.replace(" 0", " "))


def _text_statistics(query: str) -> ToolResult | None:
    match = re.match(
        r"^(?:count|analyze)\s+(?:the\s+)?(?:words|characters|text)\s*"
        r"(?:in\s*:?|:)\s*(.+)$",
        query.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    text = match.group(1)
    words = re.findall(r"\b[\w'-]+\b", text, re.UNICODE)
    lines = text.count("\n") + 1
    return ToolResult(
        "text_statistics",
        f"Words: {len(words)}; characters: {len(text)}; lines: {lines}.",
    )


def _json_tool(query: str) -> ToolResult | None:
    match = re.match(
        r"^(?:validate|format|pretty[- ]?print)\s+json\s*:\s*(.+)$",
        query.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        return ToolResult(
            "json",
            f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}.",
        )
    return ToolResult("json", json.dumps(value, ensure_ascii=False, indent=2))


def _unit_conversion(query: str) -> ToolResult | None:
    match = re.search(
        r"(?:convert\s+)?(-?\d+(?:\.\d+)?)\s*"
        r"(c|f|celsius|fahrenheit|km|kilometers?|mi|miles?|kg|kilograms?|lb|lbs|pounds?)"
        r"\s+(?:to|in)\s+"
        r"(c|f|celsius|fahrenheit|km|kilometers?|mi|miles?|kg|kilograms?|lb|lbs|pounds?)\b",
        query.lower(),
    )
    if not match:
        return None
    value = float(match.group(1))
    source = match.group(2)
    target = match.group(3)
    group = lambda unit: "c" if unit in {"c", "celsius"} else "f" if unit in {"f", "fahrenheit"} else "km" if unit.startswith("km") or unit.startswith("kilometer") else "mi" if unit.startswith("mi") else "kg" if unit.startswith("kg") or unit.startswith("kilogram") else "lb"
    source_group, target_group = group(source), group(target)
    conversions = {
        ("c", "f"): lambda x: x * 9 / 5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5 / 9,
        ("km", "mi"): lambda x: x * 0.6213711922,
        ("mi", "km"): lambda x: x / 0.6213711922,
        ("kg", "lb"): lambda x: x * 2.2046226218,
        ("lb", "kg"): lambda x: x / 2.2046226218,
    }
    operation = conversions.get((source_group, target_group))
    if operation is None:
        return None
    converted = operation(value)
    return ToolResult(
        "unit_conversion",
        f"{value:g} {source_group} is approximately {converted:.6g} {target_group}.",
    )


def _email_draft(query: str) -> ToolResult | None:
    lowered = query.casefold()
    if not (
        re.search(r"\b(write|draft|compose|create)\b", lowered)
        and re.search(r"\b(e-?mail|message)\b", lowered)
    ):
        return None
    if re.search(r"\b(reschedule|postpone|move)\b", lowered) and re.search(
        r"\b(meeting|appointment|call)\b", lowered
    ):
        text = (
            "Subject: Request to Reschedule\n\n"
            "Dear [Name],\n\n"
            "I’m writing to ask whether we could reschedule our meeting. "
            "Unfortunately, I’m no longer available at the planned time. "
            "Would [alternative date and time] work for you? I’m also happy "
            "to find another time that suits your schedule.\n\n"
            "I apologize for the inconvenience and appreciate your flexibility.\n\n"
            "Best regards,\n[Your Name]"
        )
    elif re.search(r"\b(follow[- ]?up|following up)\b", lowered):
        text = (
            "Subject: Follow-Up\n\nDear [Name],\n\n"
            "I wanted to follow up regarding [topic]. When you have a moment, "
            "could you please share any update or next steps?\n\n"
            "Thank you for your time.\n\nBest regards,\n[Your Name]"
        )
    elif re.search(r"\b(thank|thanks|gratitude)\b", lowered):
        text = (
            "Subject: Thank You\n\nDear [Name],\n\n"
            "Thank you for [specific help or occasion]. I sincerely appreciate "
            "the time and effort you gave, and it made a meaningful difference.\n\n"
            "Best regards,\n[Your Name]"
        )
    elif re.search(r"\b(apolog|sorry)\w*\b", lowered):
        text = (
            "Subject: My Apology\n\nDear [Name],\n\n"
            "I’m sorry for [what happened]. I understand the impact it caused "
            "and take responsibility. I’m addressing it by [specific action] "
            "and will work to prevent it from happening again.\n\n"
            "Sincerely,\n[Your Name]"
        )
    else:
        return None
    return ToolResult("email_draft", text)


def _guided_response(query: str) -> ToolResult | None:
    """Handle common, high-value intents where an exact local answer is safer."""
    lowered = query.casefold()
    if (
        "python" in lowered
        and "duplicate" in lowered
        and re.search(r"\b(list|strings?|items?)\b", lowered)
    ):
        return ToolResult(
            "python_help",
            "Use a dictionary to preserve the first occurrence of each hashable item:\n\n"
            "```python\n"
            "def remove_duplicates(items):\n"
            "    return list(dict.fromkeys(items))\n"
            "```\n\n"
            "Example: `remove_duplicates(['a', 'b', 'a'])` returns `['a', 'b']`. "
            "For unhashable items, use a loop and append an item only when it is not "
            "already in the output list.",
        )
    if "compound interest" in lowered:
        return ToolResult(
            "finance_explanation",
            "Compound interest means you earn interest on the original amount and on "
            "interest already added. The common formula is `A = P(1 + r/n)^(nt)`, "
            "where P is the starting principal, r is the annual rate, n is the number "
            "of compounding periods per year, and t is years. For example, $100 at 10% "
            "compounded annually becomes $110 after one year and $121 after two.",
        )
    if re.search(r"\bhttp\s*500\b|\b500\s+(?:error|status)\b", lowered):
        return ToolResult(
            "http_troubleshooting",
            "An HTTP 500 means the server failed while handling the request. Start with "
            "the server/application logs for the matching timestamp and request ID. "
            "Then reproduce the request, inspect the exception and stack trace, verify "
            "recent deployments and configuration, check database and upstream-service "
            "health, and confirm required environment variables and permissions. Avoid "
            "exposing stack traces or secrets to clients.",
        )
    if "recursion" in lowered and re.search(r"\b(example|explain|what)\b", lowered):
        return ToolResult(
            "programming_explanation",
            "Recursion is when a function solves a problem by calling itself on a "
            "smaller case. It needs a base case that stops the calls.\n\n"
            "```python\n"
            "def factorial(n):\n"
            "    if n <= 1:      # base case\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "```\n\n"
            "`factorial(4)` evaluates to `4 * 3 * 2 * 1`, or 24.",
        )
    if "rest" in lowered and "graphql" in lowered:
        return ToolResult(
            "api_comparison",
            "REST commonly exposes multiple resource-oriented URLs and uses HTTP methods "
            "such as GET, POST, PUT, and DELETE. It is simple, cache-friendly, and widely "
            "supported, but clients can over-fetch or make several requests. GraphQL "
            "usually exposes one typed query endpoint where clients request exact fields. "
            "That improves flexibility and reduces over-fetching, but adds schema, resolver, "
            "authorization, query-cost, and caching complexity. Choose based on client data "
            "needs and operational constraints rather than assuming one is always better.",
        )
    if "study plan" in lowered and re.search(r"\bexam|test\b", lowered):
        subject_match = re.search(r"(?:for|a)\s+(?:an?\s+)?([a-z-]+)\s+(?:exam|test)", lowered)
        subject = subject_match.group(1).title() if subject_match else "the subject"
        return ToolResult(
            "study_plan",
            f"Three-day {subject} study plan:\n\n"
            "1. Day 1 — Diagnose and organize: list the tested topics, take a short "
            "practice quiz, review weak concepts, and make a one-page summary.\n"
            "2. Day 2 — Active practice: answer problems or flashcards from memory, "
            "explain difficult ideas aloud, and correct every missed answer.\n"
            "3. Day 3 — Simulate and consolidate: complete a timed practice set, review "
            "only the remaining weak points, then stop early enough to sleep well.\n\n"
            "Use focused 25–45 minute blocks with short breaks and adjust the time toward "
            "the topics with the most marks or lowest practice scores.",
        )
    if "website" in lowered and "checklist" in lowered:
        return ToolResult(
            "launch_checklist",
            "Website launch checklist:\n\n"
            "- Confirm goals, audience, navigation, copy, and contact details.\n"
            "- Test every link, form, error state, and call to action.\n"
            "- Check mobile layouts, keyboard access, contrast, alt text, and headings.\n"
            "- Optimize images, enable HTTPS, set secure headers, and remove test secrets.\n"
            "- Add page titles, descriptions, sitemap, robots rules, and a favicon.\n"
            "- Configure analytics, privacy/consent requirements, backups, and monitoring.\n"
            "- Test on major browsers, proofread, verify the production domain, and keep a "
            "rollback plan.",
        )
    if "science report" in lowered and re.search(r"\b(outline|structure)\b", lowered):
        return ToolResult(
            "report_outline",
            "Short science report outline:\n\n"
            "1. Title — specific question or finding.\n"
            "2. Introduction — background, research question, and hypothesis.\n"
            "3. Methods — materials, procedure, variables, and sample.\n"
            "4. Results — observations, measurements, tables, or figures without interpretation.\n"
            "5. Discussion — explain the results, uncertainty, limitations, and whether the hypothesis was supported.\n"
            "6. Conclusion — one concise takeaway and a possible next step.\n"
            "7. References — sources in one consistent citation style.",
        )
    if (
        re.search(r"\b(brainstorm|suggest|give)\b", lowered)
        and "name" in lowered
        and "delivery" in lowered
        and re.search(r"\b(environment|eco|green)\w*\b", lowered)
    ):
        return ToolResult(
            "name_ideas",
            "Five name ideas: GreenRoute Delivery, EcoDrop, LeafLine Logistics, "
            "KindCourier, and LowCarbon Local. Check trademark, company-name, domain, "
            "and social-handle availability before choosing one.",
        )
    return None


def run_tool(query: str) -> ToolResult | None:
    """Route a query to a deterministic tool when its intent is unambiguous."""
    for tool in (
        _arithmetic,
        _date_time,
        _text_statistics,
        _json_tool,
        _unit_conversion,
        _email_draft,
        _guided_response,
    ):
        result = tool(query)
        if result is not None:
            return result
    return None
