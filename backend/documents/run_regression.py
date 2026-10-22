"""Regression run of regression_questions.csv against the document chatbot (document mode).

Run from the backend folder:
    python -m documents.run_regression                 # in-process, no server needed
    python -m documents.run_regression --http http://localhost:8000   # via the running API
    python -m documents.run_regression --only 3 7 24   # a subset of question numbers
    python -m documents.run_regression --questions documents/regression_questions_derivations.json

Writes to documents/regression_output/<timestamp>/:
    results.csv   one row per question (answer, sources, score, status, timing)
    results.json  the same with full detail
    summary.md    pass/fail overview

Scoring is a heuristic: every number in the expected answer is looked for in the bot's answer
(commas ignored, 0.5% tolerance). PASS >= 80% found, PARTIAL >= 40%, otherwise FAIL.
Rows whose expected answer says the bot must not invent something are marked REVIEW for a
human check. Always read the answers of PARTIAL / FAIL / REVIEW rows yourself.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "regression_questions.csv"
OUTPUT_ROOT = HERE / "regression_output"

NUM_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> list[float]:
    values = []
    for m in NUM_RE.finditer(text or ""):
        raw = m.group().replace(",", "").lstrip("+")
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def score(expected: str, answer: str) -> tuple[float, list[float], list[float]]:
    want = numbers(expected)
    got = numbers(answer)
    # Compare magnitudes too: the bot may print a decline as "6.7%" instead of "-6.7%".
    got_abs = [abs(g) for g in got]
    found, missing = [], []
    for w in want:
        ok = any(abs(g - w) <= max(0.005, abs(w) * 0.005) for g in got) or \
             any(abs(g - abs(w)) <= max(0.005, abs(w) * 0.005) for g in got_abs)
        (found if ok else missing).append(w)
    return (len(found) / len(want) if want else 0.0), found, missing


def status_for(expected: str, ratio: float, answer: str) -> str:
    if not answer.strip():
        return "FAIL"
    if not expected.strip():
        return "UNSCORED"  # no expected answer supplied: read the bot's answer yourself
    if re.search(r"should not|must not|no reason", expected, re.I):
        return "REVIEW"
    return "PASS" if ratio >= 0.8 else "PARTIAL" if ratio >= 0.4 else "FAIL"


def ask_in_process(question: str) -> tuple[str, list[dict], str]:
    from . import service  # imported lazily so --http mode does not load the models

    answer, sources, error = [], [], ""
    for event in service.stream_answer(question):
        try:
            payload = json.loads(event[len("data: "):].strip())
        except (ValueError, IndexError):
            continue
        kind = payload.get("type")
        if kind == "token":
            answer.append(payload.get("content", ""))
        elif kind == "sources":
            sources = payload.get("sources", [])
        elif kind == "error":
            error = payload.get("content", "error")
    return "".join(answer), sources, error


def ask_http(base_url: str, question: str) -> tuple[str, list[dict], str]:
    import httpx

    answer, sources, error = [], [], ""
    with httpx.stream("POST", base_url.rstrip("/") + "/chat/stream",
                      json={"query": question, "source": "documents"}, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[5:].strip())
            except ValueError:
                continue
            kind = payload.get("type")
            if kind == "token":
                answer.append(payload.get("content", ""))
            elif kind == "sources":
                sources = payload.get("sources", [])
            elif kind == "error":
                error = payload.get("content", "error")
    return "".join(answer), sources, error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http", metavar="URL", help="call a running backend instead of in-process")
    parser.add_argument("--only", type=int, nargs="*", help="question numbers to run")
    parser.add_argument("--questions", default=str(QUESTIONS))
    args = parser.parse_args()

    if args.questions.lower().endswith(".json"):  # same fields as the CSV, plus an optional note
        rows = [{k: str(v) for k, v in item.items()}
                for item in json.loads(Path(args.questions).read_text(encoding="utf-8-sig"))]
    else:
        with open(args.questions, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if args.only:
        rows = [r for r in rows if int(r["question_number"]) in set(args.only)]

    out_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    ask = (lambda q: ask_http(args.http, q)) if args.http else ask_in_process

    results = []
    for row in rows:
        number, question = row["question_number"], row["question"]
        print(f"[{number}/{len(rows) if not args.only else '*'}] {question}", flush=True)
        started = time.perf_counter()
        try:
            answer, sources, error = ask(question)
        except Exception as exc:  # keep going; one bad question must not stop the run
            answer, sources, error = "", [], f"{type(exc).__name__}: {exc}"
        elapsed = round(time.perf_counter() - started, 2)
        ratio, found, missing = score(row["expected_answer"], answer)
        status = "ERROR" if error and not answer else status_for(row["expected_answer"], ratio, answer)
        cited = sorted({s.get("subject", "") for s in sources if s.get("subject")})
        results.append({
            "question_number": number, "question": question,
            "expected_answer": row["expected_answer"], "expected_source": row["source"],
            "bot_answer": answer.strip(), "bot_sources": " | ".join(cited),
            "numbers_found": f"{len(found)}/{len(found) + len(missing)}",
            "missing_numbers": ", ".join(f"{m:g}" for m in missing),
            "status": status, "seconds": elapsed, "error": error,
        })
        print(f"    -> {status} ({results[-1]['numbers_found']} numbers, {elapsed}s)", flush=True)

    fields = list(results[0].keys()) if results else []
    with open(out_dir / "results.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = [f"# Document-mode regression run {out_dir.name}", "",
             f"Mode: {'HTTP ' + args.http if args.http else 'in-process'}  |  Questions: {len(results)}", "",
             "| Status | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "| # | Status | Numbers | Seconds | Question |", "|---|---|---|---|---|"]
    lines += [f"| {r['question_number']} | {r['status']} | {r['numbers_found']} | {r['seconds']} | {r['question']} |"
              for r in results]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
