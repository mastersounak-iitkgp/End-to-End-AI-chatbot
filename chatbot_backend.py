"""
Persistent Interactive Chatbot with LangGraph, Mistral AI & Tool Calling
========================================================================
Architecture: START -> chat -> [tools_condition] -> tools -> chat -> END
Model: Mistral AI (ministral-8b-latest, 128k context window, free tier)
Tools:
  1) duckduckgo_search   — Live web search via DuckDuckGo
  2) calculate_math      — Reliable AST-based mathematical evaluation
  3) check_stock_price   — Real-time financial market quotes via yfinance
Persistence: SQLite checkpointer (SqliteSaver) with per-thread_id isolation
"""

import os
import sys
import ast
import math
import operator
import warnings
from typing import TypedDict, Annotated, Optional, Generator, Any
from dotenv import load_dotenv

# Suppress noisy Pandas / yfinance deprecation warnings in stdout
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_engine import retrieve_relevant_chunks, get_thread_doc_info

# Ensure UTF-8 output encoding for terminals supporting unicode/emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 1. Load API keys from .env
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo for live facts, current events, sports scores, recent news, and real-time information.
    Use this tool whenever the user asks about live scores (e.g. cricket, football), recent news, weather, or anything requiring up-to-date web knowledge.
    """
    clean_query = query.strip().strip('"').strip("'")
    if not clean_query:
        return "Search Error: Empty query provided."

    try:
        # Import DDGS (supports either ddgs or duckduckgo_search)
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(clean_query, max_results=4))

        if not results:
            return (
                f"No web search results were found for: '{clean_query}'. "
                "Please answer using your existing knowledge or explain politely that live search returned no matches."
            )

        formatted = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            snippet = item.get("body", "").strip()
            link = item.get("href", "")
            formatted.append(f"[{i}] {title}\nSummary: {snippet}\nURL: {link}")

        return "\n\n".join(formatted)

    except Exception as e:
        return (
            f"Note: Web search for '{clean_query}' could not be completed at this moment ({str(e)}). "
            "Please safely answer based on your available knowledge without throwing an error."
        )


# Safe AST Math Evaluator Definitions
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "cbrt": getattr(math, "cbrt", lambda x: x ** (1 / 3)),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": getattr(math, "lcm", lambda a, b: abs(a * b) // math.gcd(a, b)),
    "degrees": math.degrees,
    "radians": math.radians,
}


def _eval_ast_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Operator {op_type.__name__} is not allowed.")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unary operator {op_type.__name__} is not allowed.")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct mathematical functions are allowed.")
        func_name = node.func.id.lower()
        if func_name in ALLOWED_FUNCTIONS:
            args = [_eval_ast_node(arg) for arg in node.args]
            return ALLOWED_FUNCTIONS[func_name](*args)
        raise ValueError(f"Unknown or unsupported math function '{func_name}'.")
    elif isinstance(node, ast.Name):
        name = node.id.lower()
        if name in ALLOWED_CONSTANTS:
            return ALLOWED_CONSTANTS[name]
        raise ValueError(f"Unknown mathematical variable or constant '{node.id}'.")
    else:
        raise ValueError(f"Unsupported mathematical syntax: {type(node).__name__}")


@tool
def calculate_math(expression: str) -> str:
    """
    A reliable calculator tool to evaluate mathematical functions and arithmetic expressions.
    Supports addition (+), subtraction (-), multiplication (*), division (/), powers (^ or **), modulo (%),
    as well as math functions: sqrt, cbrt, sin, cos, tan, log, exp, abs, factorial, round, ceil, floor, and constants pi, e.
    Use this tool whenever exact numerical calculations are required to ensure 100% precision without hallucinations.
    """
    clean_expr = expression.strip()
    if not clean_expr:
        return "Calculator Error: Empty mathematical expression provided."

    try:
        normalized = clean_expr.replace("^", "**")
        parsed = ast.parse(normalized, mode="eval")
        result = _eval_ast_node(parsed.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{clean_expr} = {result}"
    except ZeroDivisionError:
        return f"Math Error: Division by zero in expression '{clean_expr}'."
    except OverflowError:
        return f"Math Error: Numerical overflow in expression '{clean_expr}'."
    except Exception as e:
        return f"Math Error evaluating '{clean_expr}': {str(e)}"


@tool
def check_stock_price(ticker: str) -> str:
    """
    Check the latest stock price, daily price change, and market valuation for any ticker symbol (e.g. AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, BTC-USD, RELIANCE.NS).
    Use this tool whenever the user asks for stock quotes, market prices, or company financial data.
    """
    symbol = ticker.strip().upper().replace("$", "")
    if not symbol:
        return "Stock Error: Empty ticker symbol provided."

    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        fi = t.fast_info

        price = fi.last_price
        if price is None:
            # Fallback to recent history
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
            else:
                return f"Could not retrieve stock data for ticker '{symbol}'. Please ensure it is a valid stock or crypto symbol."

        curr = fi.currency or "USD"
        prev_close = fi.previous_close
        change_pct_str = "N/A"
        if prev_close and prev_close > 0:
            change_pct = ((price - prev_close) / prev_close) * 100
            change_pct_str = f"{change_pct:+.2f}%"

        day_high = fi.day_high
        day_low = fi.day_low
        mkt_cap = fi.market_cap

        # Format market cap
        mkt_cap_str = "N/A"
        if mkt_cap:
            if mkt_cap >= 1e12:
                mkt_cap_str = f"${mkt_cap / 1e12:.2f} Trillion"
            elif mkt_cap >= 1e9:
                mkt_cap_str = f"${mkt_cap / 1e9:.2f} Billion"
            elif mkt_cap >= 1e6:
                mkt_cap_str = f"${mkt_cap / 1e6:.2f} Million"
            else:
                mkt_cap_str = f"${mkt_cap:,.0f}"

        lines = [
            f"Symbol: {symbol}",
            f"Current Price: {price:,.2f} {curr}",
            f"Daily Change: {change_pct_str}",
            f"Previous Close: {prev_close:,.2f} {curr}" if prev_close else None,
            f"Day Range: {day_low:,.2f} - {day_high:,.2f} {curr}" if (day_low and day_high) else None,
            f"Market Cap: {mkt_cap_str}",
        ]
        return "\n".join(l for l in lines if l)

    except Exception as e:
        return f"Unable to fetch stock data for '{symbol}' ({str(e)}). Please verify the ticker."


@tool
def search_uploaded_documents(query: str, config: RunnableConfig) -> str:
    """
    Search the uploaded PDF document for relevant information, facts, data, section clauses, or answers.
    Use this tool whenever the user asks questions about their uploaded document or PDF.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    doc_info = get_thread_doc_info(thread_id)
    if not doc_info:
        return (
            "No document has been uploaded for this conversation yet. "
            "Please inform the user to upload their PDF document using the file uploader in the sidebar."
        )

    clean_query = query.strip()
    if not clean_query:
        return "Search Error: Empty query provided for document search."

    try:
        docs, formatted_context = retrieve_relevant_chunks(clean_query, thread_id, final_top_k=5)
        if not docs:
            return f"No relevant content was found in '{doc_info.get('file_name', 'document')}' for the query '{clean_query}'."
        return formatted_context
    except Exception as e:
        return f"Error retrieving document content ({str(e)}). Please answer based on general knowledge."


# List of all available tools
TOOLS = [duckduckgo_search, calculate_math, check_stock_price, search_uploaded_documents]
tool_node = ToolNode(TOOLS)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. State Definition
# ═══════════════════════════════════════════════════════════════════════════════

class ChatState(TypedDict):
    """
    Chat state schema.
    Uses add_messages reducer so each turn is automatically appended to history.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Initialize Model & Bind Tools
# ═══════════════════════════════════════════════════════════════════════════════

base_model = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0.3,
    streaming=True,
    max_retries=5,
    timeout=60,
)
model_with_tools = base_model.bind_tools(TOOLS)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Define Chat Node
# ═══════════════════════════════════════════════════════════════════════════════

def chat(state: ChatState):
    """
    Invokes the Mistral model bound with tools.
    Decides whether to respond directly or generate tool calls.
    """
    messages = list(state["messages"])

    # Prepend optional system prompt if provided and not yet present in messages
    if state.get("system_prompt") and not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=state["system_prompt"])] + messages

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Build and Compile Graph with ToolNode & Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def create_chatbot_app(db_path: str = ":memory:"):
    """
    Builds and compiles the LangGraph workflow:
      START -> chat -> [tools_condition]
                         ├─ (tool_calls) ─> tools -> chat
                         └─ (no calls)   ─> END

    Args:
        db_path: Path to the SQLite database file for conversation checkpointing.
    """
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Streaming Helper for External Callers (Streamlit frontend & CLI)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_chat_response(app, payload: dict, config: dict) -> Generator[dict, None, None]:
    """
    Generator yielding structured streaming events from the LangGraph execution:
      - {"type": "tool_call", "name": ..., "args": ..., "id": ...}
      - {"type": "tool_result", "name": ..., "output": ..., "id": ...}
      - {"type": "token", "content": ...}

    Args:
        app:     Compiled LangGraph application.
        payload: Dict with at least {"messages": [...]} and optional "system_prompt".
        config:  LangGraph config dict, e.g. {"configurable": {"thread_id": "..."}}
    """
    seen_tool_calls = set()
    for chunk, _metadata in app.stream(payload, config=config, stream_mode="messages"):
        # 1. Tool Call Chunk (Emitted by chat node when delegating to a tool)
        if getattr(chunk, "tool_calls", None):
            for tc in chunk.tool_calls:
                call_id = tc.get("id") or tc.get("name")
                if tc.get("name") and call_id not in seen_tool_calls:
                    seen_tool_calls.add(call_id)
                    yield {
                        "type": "tool_call",
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                        "id": tc.get("id"),
                    }

        # 2. Tool Execution Result (Emitted by tools node)
        elif getattr(chunk, "type", None) == "tool":
            yield {
                "type": "tool_result",
                "name": getattr(chunk, "name", "tool"),
                "output": str(chunk.content),
                "id": getattr(chunk, "tool_call_id", None),
            }

        # 3. AI Text Tokens (Emitted when synthesizing the answer)
        elif isinstance(chunk, AIMessageChunk):
            content = chunk.content
            if isinstance(content, str) and content:
                yield {"type": "token", "content": content}
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            yield {"type": "token", "content": text}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Interactive Indefinite CLI Loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_chatbot(app, thread_id: str = "my-chat-session", system_prompt: Optional[str] = None):
    """
    Runs an interactive chatbot loop in the terminal with persistence and tool calling.
    """
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 65)
    print("🤖 Mistral AI Chatbot Started! (Tools: Search, Calculator, Stocks)")
    print("Commands: Type 'exit', 'quit', or 'bye' to close.")
    print("=" * 65)

    first_turn = True

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                print("\n🤖 Bot: Goodbye! Have a great day.")
                break

            payload = {"messages": [HumanMessage(content=user_input)]}
            if first_turn and system_prompt:
                payload["system_prompt"] = system_prompt
                first_turn = False

            print("\nBot: ", end="", flush=True)
            for event in stream_chat_response(app, payload, config):
                ev_type = event.get("type")
                if ev_type == "tool_call":
                    print(f"\n⚙️  [Calling {event['name']}({event.get('args', {})})]", flush=True)
                elif ev_type == "tool_result":
                    print(f"   ✓ [Tool Completed]\nBot: ", end="", flush=True)
                elif ev_type == "token":
                    print(event["content"], end="", flush=True)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user.")
            break
        except Exception as e:
            print(f"\n⚠️ Error occurred: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLI Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_chatbot_app(db_path="chatbot_cli.db")
    run_chatbot(
        app,
        thread_id="my-cli-session",
        system_prompt="You are a helpful AI assistant equipped with web search, calculator, and stock checking tools."
    )
