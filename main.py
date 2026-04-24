import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from load_env import openai_key  # noqa: E402
from query_rag import two_stage_search  # noqa: E402


def _print_retrieval(label: str, rows, key_title: str):
    print(f"\n--- {label} ---")
    if not rows:
        print("(none)")
        return
    for i, row in enumerate(rows, 1):
        d = row.get("distance")
        dist = f" distance={d:.4f}" if d is not None else ""
        print(f"\n[{i}] {row.get('id', '')} {row.get(key_title, '')}{dist}")
        text = row.get("text", "")
        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(preview)


def run_query(q: str, debug: bool, use_llm: bool):
    out = two_stage_search(q)
    policies = out["policies"]
    playbooks = out["playbooks"]

    if debug:
        _print_retrieval("Stage 1: policies", policies, "title")
        _print_retrieval("Stage 2: playbooks", playbooks, "title")
    else:
        print("Top matches: policy titles:", ", ".join(p.get("title", "") for p in policies))
        print(
            "Top matches: playbook titles:",
            ", ".join(p.get("title", "") for p in playbooks),
        )

    if use_llm:
        if not openai_key():
            print(
                "\nLLM mode requested but OPENAI_API_KEY is missing. "
                "Copy .env.example to .env and set your key."
            )
            return
        from llm import draft_answer  # noqa: E402

        text = draft_answer(q, policies, playbooks)
        print("\n--- Suggested answer (LLM, uses only retrieved context) ---\n")
        print(text)
    else:
        if not debug:
            print(
                "\n(Retrieval only. Use --llm and a .env file for an optional OpenAI answer.)"
            )


def main():
    p = argparse.ArgumentParser(
        description="Toy two-stage RAG on a fake coffee wholesaler helpdesk (policies, then playbooks)."
    )
    p.add_argument("question", nargs="*", help="Question (interactive if omitted).")
    p.add_argument(
        "--debug", action="store_true", help="Print full retrieved chunks."
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Call OpenAI to draft an answer (requires OPENAI_API_KEY in .env).",
    )
    args = p.parse_args()

    if args.question:
        run_query(" ".join(args.question), args.debug, args.llm)
        return

    print("Aurora Bean Roasters, toy RAG demo (type 'exit' to quit).")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        run_query(line, args.debug, args.llm)


if __name__ == "__main__":
    main()
