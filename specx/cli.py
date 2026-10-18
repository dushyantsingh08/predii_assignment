"""Command line interface:  python -m specx <command> ...

  ingest     parse the PDF, chunk, embed, cache
  query      answer one question
  batch      answer many questions (one per line) into a single CSV/JSON
  eval       score the pipeline against eval/gold.json
  inspect    show how a PDF page was parsed (debugging aid)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .pipeline import Pipeline


def _settings(args) -> Settings:
    s = Settings()
    if getattr(args, "pdf", None):
        s.pdf_path = args.pdf
    if getattr(args, "cache", None):
        s.cache_dir = args.cache
    if getattr(args, "embed", None):
        s.embed_backend = args.embed
    if getattr(args, "llm", None):
        s.llm_provider = args.llm
    if getattr(args, "model", None):
        s.llm_model = args.model
    if getattr(args, "k", None):
        s.top_k = args.k
    return s


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pdf", help="path to the service manual PDF (default: $SPECX_PDF)")
    p.add_argument("--cache", help="cache directory (default: .specx_cache)")
    p.add_argument("--embed", choices=["sentence-transformers", "openai", "tfidf"], help="embedding backend")
    p.add_argument("--llm", choices=["auto", "anthropic", "openai", "none"], help="LLM provider (none = rule-based tables only)")
    p.add_argument("--model", help="LLM model name")
    p.add_argument("-k", type=int, help="number of chunks to retrieve (default 6)")


def _minimal(result_json: str) -> str:
    data = json.loads(result_json)
    keep = ("component", "spec_type", "value", "unit")
    return json.dumps([{k: r.get(k) for k in keep} for r in data], indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="specx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="parse + chunk + embed the PDF")
    _add_common(p)
    p.add_argument("--force", action="store_true", help="re-parse the PDF even if cached")

    p = sub.add_parser("query", help="ask one question")
    _add_common(p)
    p.add_argument("question")
    p.add_argument("--mode", choices=["auto", "llm", "rules"], default="auto")
    p.add_argument("--format", choices=["json", "csv"], default="json")
    p.add_argument("--minimal", action="store_true", help="JSON with only component/spec_type/value/unit")
    p.add_argument("--trace", action="store_true", help="include retrieval trace in JSON output")
    p.add_argument("--strict", action="store_true", help="drop records whose value was not found in the source text")
    p.add_argument("-o", "--out", help="write to file instead of stdout")

    p = sub.add_parser("batch", help="answer many questions")
    _add_common(p)
    p.add_argument("file", help="text file with one question per line")
    p.add_argument("--mode", choices=["auto", "llm", "rules"], default="auto")
    p.add_argument("--format", choices=["json", "csv"], default="csv")
    p.add_argument("--strict", action="store_true")
    p.add_argument("-o", "--out", required=True)

    p = sub.add_parser("eval", help="evaluate against the gold set")
    _add_common(p)
    p.add_argument("--gold", default="eval/gold.json")
    p.add_argument("--mode", choices=["auto", "llm", "rules"], default="rules")

    p = sub.add_parser("inspect", help="show how a page was parsed")
    _add_common(p)
    p.add_argument("page", type=int, help="1-based PDF page number")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    s = _settings(args)

    if args.cmd == "inspect":
        from .pdf_loader import load_manual

        manual = load_manual(s.pdf_path, s, pages=range(args.page - 1, args.page), progress=False)
        for d in manual.docs:
            print(f"# {d.section_label} | {d.doc_type} | file {d.file_id} | rev {d.revision}")
            for e in d.elements:
                tag = f"table:{e.table_kind}" if e.kind == "table" else (f"heading{e.level}" if e.kind == "heading" else "text")
                print(f"--- [{tag}] p{e.page}\n{e.text}")
        return 0

    pipe = Pipeline(s)

    if args.cmd == "ingest":
        pipe.build(force=args.force)
        print(f"Indexed {len(pipe.chunks)} chunks from {len(pipe.manual.docs)} documents into {pipe.cache}/")
        return 0

    pipe.ensure_ready()

    if args.cmd == "query":
        res = pipe.query(args.question, mode=args.mode, strict=args.strict)
        for w in res.warnings:
            print(f"warning: {w}", file=sys.stderr)
        out = res.to_csv() if args.format == "csv" else res.to_json(full=args.trace)
        if args.format == "json" and args.minimal and not args.trace:
            out = _minimal(out)
        if args.out:
            Path(args.out).write_text(out, encoding="utf-8")
        else:
            print(out)
        return 0

    if args.cmd == "batch":
        questions = [q.strip() for q in Path(args.file).read_text(encoding="utf-8").splitlines() if q.strip() and not q.startswith("#")]
        rows, jsons = [], []
        for q in questions:
            res = pipe.query(q, mode=args.mode, strict=args.strict)
            for r in res.records:
                rows.append({"query": q, **r.to_flat_dict()})
            jsons.append({"query": q, "mode": res.mode, "results": json.loads(res.to_json())})
            print(f"{q!r}: {len(res.records)} record(s) [{res.mode}]", file=sys.stderr)
        if args.format == "json":
            Path(args.out).write_text(json.dumps(jsons, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            import csv

            from .schema import CSV_COLUMNS

            with open(args.out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["query"] + CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {args.out}", file=sys.stderr)
        return 0

    if args.cmd == "eval":
        from .evaluate import evaluate

        rep = evaluate(pipe, args.gold, mode=args.mode)
        print(rep.summary())
        if rep.failures:
            print("\nmisses:\n  " + "\n  ".join(rep.failures))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
