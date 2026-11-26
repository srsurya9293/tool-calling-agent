"""Your agent's tools. A tool is a normal Python function with a clear
docstring and typed inputs. The model reads the docstring to decide when
to use it. Add your own tools at the bottom and list them in TOOLS.

SAFETY RULES for this assignment:
- A tool may READ your data, COMPUTE something, or DRAFT text.
- A tool must never send, delete, pay or change anything outside the app.
- Everything a tool returns goes to the model, so keep results short
  (use clip) and remember that text from your data file is untrusted."""
import csv
from datetime import date
from pathlib import Path

from langchain_core.tools import tool

DATA_FILE = Path(__file__).parent / "data" / "notes.csv"
MAX_CHARS = 600


def clip(text, limit=MAX_CHARS):
    """Cut long text so one tool result cannot flood the model."""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ...(cut)"


def read_rows():
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@tool
def today() -> str:
    """Return today's date. Use it when the answer depends on the current date."""
    return date.today().isoformat()


@tool
def lookup_notes(keyword: str) -> str:
    """Find up to 3 notes whose title or text contains the keyword. Use it to look things up in the notes file."""
    word = keyword.strip().lower()
    if not word:
        return "Please give a keyword to look for."
    hits = [r for r in read_rows() if word in (r["title"] + " " + r["text"]).lower()]
    if not hits:
        return f"No notes found for '{keyword}'."
    return clip("\n".join(f"- {r['title']}: {r['text']}" for r in hits[:3]))


# ---- ADD YOUR OWN TOOLS BELOW (at least three, each with a different job) ----
# 1. one that READS your data file,
# 2. one that COMPUTES something,
# 3. one that DRAFTS text (it only returns text, it never sends).
# @tool
# def my_tool(some_input: str) -> str:
#     """One clear sentence: what it does and when to use it."""
#     return clip("...")


TOOLS = [today, lookup_notes]