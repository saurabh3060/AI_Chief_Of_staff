"""A restricted arithmetic evaluator shared by both QA pipelines, exposed to the chat model as the
`calculate` tool so it never computes arithmetic itself: numbers, + - * / // % **, parentheses,
unary minus only — no names, calls or attribute access, so it can't do anything but arithmetic."""
import ast
import operator
import re

_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d)")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> dict:
    """Commas between digits are stripped before parsing: the model routinely copies a number
    straight out of as_printed ("-40,842"), and since no supported syntax here ever needs a comma
    (no calls, no tuples, no lists), a digit-flanked comma can only ever be a thousands separator."""
    try:
        tree = ast.parse(_THOUSANDS_SEP.sub("", expression), mode="eval")
        return {"result": _eval_node(tree.body)}
    except Exception as e:
        return {"error": f"invalid expression: {e}"}


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate one arithmetic expression built from values you extracted "
                        "from the context. Always use this for arithmetic; never compute it yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "description": "e.g. '1234.5 - 987.6' or '(500-450)/450*100'."},
            },
            "required": ["expression"],
        },
    },
}
