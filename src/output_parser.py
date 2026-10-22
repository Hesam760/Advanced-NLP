import json
import re


def parse_model_answer(raw_output):

    if isinstance(raw_output, dict):
        text = raw_output.get("answer", "")
        if not isinstance(text, str):
            return ""
    elif isinstance(raw_output, str):
        text = raw_output
    else:
        return ""

    text = text.strip()

    if not text:
        return ""

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""

        if not isinstance(payload, dict):
            return ""

        text = payload.get("answer", "")
        if not isinstance(text, str):
            return ""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    while lines:
        match = re.match(
            r"^(?:final answer|answer)\s*:\s*(.*)$",
            lines[0],
            flags=re.IGNORECASE,
        )

        if not match:
            break

        value = match.group(1).strip()

        if value:
            lines[0] = value
            break

        lines.pop(0)

    if not lines:
        return ""

    answer = lines[0].strip()

    if (
        len(answer) >= 2
        and answer[0] == answer[-1]
        and answer[0] in {"'", '"'}
    ):
        answer = answer[1:-1].strip()

    return answer