"""
Tool Manager — Safe tools for the AI assistants.
"""

from __future__ import annotations
import math, ast, operator
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class ToolResult:
    success: bool
    output: str
    tool_name: str
    error: str = ""


SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
SAFE_FNS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log10": math.log10,
    "abs": abs, "round": round, "pi": math.pi, "e": math.e,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in SAFE_OPS:
            raise ValueError(f"Unsupported op: {op.__name__}")
        return SAFE_OPS[op](_eval_node(node.left), _eval_node(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in SAFE_OPS:
            raise ValueError(f"Unsupported op: {op.__name__}")
        return SAFE_OPS[op](_eval_node(node.operand))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FNS:
        fn = SAFE_FNS[node.func.id]
        if callable(fn):
            return fn(*[_eval_node(a) for a in node.args])
    elif isinstance(node, ast.Name) and node.id in SAFE_FNS and not callable(SAFE_FNS[node.id]):
        return SAFE_FNS[node.id]
    raise ValueError(f"Unsupported: {type(node).__name__}")


def safe_calculate(expression: str) -> ToolResult:
    try:
        expr = expression.strip()
        result = _eval_node(ast.parse(expr, mode="eval").body)
        fmt = str(int(result)) if isinstance(result, float) and result == int(result) else f"{result:.6g}"
        return ToolResult(True, f"{expr} = {fmt}", "calculator")
    except Exception as e:
        return ToolResult(False, "", "calculator", str(e))


def get_datetime(query: str = "") -> ToolResult:
    now = datetime.now()
    utc = datetime.now(timezone.utc)
    info = (
        f"Local: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"UTC: {utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Day: {now.strftime('%A')}\n"
        f"TZ: UTC{now.astimezone().strftime('%z')}"
    )
    return ToolResult(True, info, "datetime")


def convert_units(query: str) -> ToolResult:
    convs = {
        ("km","miles"): lambda x: x*0.621371, ("miles","km"): lambda x: x*1.60934,
        ("kg","lbs"): lambda x: x*2.20462, ("lbs","kg"): lambda x: x*0.453592,
        ("celsius","fahrenheit"): lambda x: x*9/5+32, ("fahrenheit","celsius"): lambda x: (x-32)*5/9,
        ("m","ft"): lambda x: x*3.28084, ("ft","m"): lambda x: x*0.3048,
        ("cm","inches"): lambda x: x*0.393701, ("inches","cm"): lambda x: x*2.54,
    }
    try:
        parts = query.lower().strip().split()
        val, fr, to = float(parts[0]), parts[1], parts[-1]
        if (fr, to) in convs:
            r = convs[(fr, to)](val)
            return ToolResult(True, f"{val:.4g} {fr} = {r:.4g} {to}", "unit_converter")
        return ToolResult(False, "", "unit_converter", f"Unsupported: {fr}→{to}")
    except Exception as e:
        return ToolResult(False, "", "unit_converter", str(e))


def web_search(query: str) -> ToolResult:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        
        if not results:
            return ToolResult(False, "", "web_search", "No results found.")
            
        formatted = "\n\n".join([f"**{r['title']}**\n{r['body']}\n*Source: {r['href']}*" for r in results])
        return ToolResult(True, formatted, "web_search")
    except ImportError:
        return ToolResult(False, "", "web_search", "duckduckgo-search not installed")
    except Exception as e:
        return ToolResult(False, "", "web_search", str(e))


TOOL_REGISTRY = {
    "calculator": {"name": "calculator", "description": "Evaluate math expressions safely.", "function": safe_calculate},
    "datetime": {"name": "datetime", "description": "Get current date/time.", "function": get_datetime},
    "unit_converter": {"name": "unit_converter", "description": "Convert between units.", "function": convert_units},
    "web_search": {"name": "web_search", "description": "Search the web for real-time information.", "function": web_search},
}


class ToolManager:
    def __init__(self):
        self.tools = dict(TOOL_REGISTRY)

    def get_tool_descriptions(self) -> str:
        return "\n".join(f"  • {t['name']}: {t['description']}" for t in self.tools.values())

    def invoke(self, tool_name: str, query: str) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(False, "", tool_name, f"Unknown tool: {tool_name}")
        return self.tools[tool_name]["function"](query)

    def detect_and_invoke(self, msg: str) -> ToolResult | None:
        m = msg.lower().strip()
        calc_triggers = ["calculate", "compute", "what is", "solve"]
        if any(t in m for t in calc_triggers):
            expr = m
            for t in calc_triggers:
                expr = expr.replace(t, "")
            expr = expr.strip().strip("?").strip()
            if expr and any(c in "0123456789+-*/()" for c in expr):
                return self.invoke("calculator", expr)
        if any(t in m for t in ["what time", "current time", "what date", "today"]):
            return self.invoke("datetime", m)
        if any(t in m for t in ["convert", "to miles", "to km", "to celsius", "to fahrenheit"]):
            return self.invoke("unit_converter", m.replace("convert", "").strip())
        if any(t in m for t in ["search for", "look up", "find information on", "who is", "what is the latest"]):
            query = m
            for t in ["search for", "look up", "find information on"]:
                query = query.replace(t, "").strip()
            return self.invoke("web_search", query)
        return None
