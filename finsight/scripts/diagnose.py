#!/usr/bin/env python3
"""
diagnose.py  —  FinSight cluster diagnostic
Checks the four things that must be true before evaluation can run:
  1. ChromaDB index exists and has data
  2. BM25 index files exist
  3. vLLM endpoint is reachable at localhost:8000
  4. V0 (LLM-only, no retrieval) actually produces an answer

Usage:
    python scripts/diagnose.py
"""

import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb_compat  # noqa: F401  — must precede any chromadb import

OK   = "  ✓"
FAIL = "  ✗"
WARN = "  !"


def check_chroma():
    print("\n[1] ChromaDB index")
    try:
        from src.utils.config_loader import load_config
        cfg = load_config()
        chroma_path = cfg["paths"]["chroma_db"]
        p = Path(chroma_path)
        if not p.exists():
            print(f"{FAIL} Directory not found: {chroma_path}")
            print("     → Run: python scripts/ingest_all.py   (or build_index.py)")
            return False

        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        coll_name = cfg["chroma"]["collection_name"]
        try:
            coll = client.get_collection(coll_name)
            n = coll.count()
            print(f"{OK}  Collection '{coll_name}' exists — {n:,} chunks")
            if n == 0:
                print(f"{WARN} Collection is empty — index may not have been built yet")
                return False
            return True
        except Exception as e:
            print(f"{FAIL} Collection '{coll_name}' not found: {e}")
            print("     → Run: python scripts/ingest_all.py   (or build_index.py)")
            return False
    except Exception as e:
        print(f"{FAIL} ChromaDB check failed: {e}")
        return False


def check_bm25():
    print("\n[2] BM25 index")
    try:
        from src.utils.config_loader import load_config
        cfg = load_config()
        bm25_path = Path(cfg["paths"]["bm25_index"])
        corpus_path = Path(cfg["paths"]["bm25_corpus"])

        ok = True
        for p in [bm25_path, corpus_path]:
            if p.exists():
                size_mb = p.stat().st_size / 1_048_576
                print(f"{OK}  {p.name}  ({size_mb:.1f} MB)")
            else:
                print(f"{FAIL} {p} not found")
                print("     → Run: python scripts/ingest_all.py   (or build_index.py)")
                ok = False
        return ok
    except Exception as e:
        print(f"{FAIL} BM25 check failed: {e}")
        return False


def check_vllm():
    print("\n[3] vLLM endpoint (localhost:8000)")
    try:
        from src.utils.config_loader import load_config
        cfg = load_config()
        base_url = cfg["generation"]["base_url"]
        models_url = base_url.rstrip("/").replace("/v1", "") + "/v1/models"

        req = urllib.request.Request(models_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            models = [m["id"] for m in body.get("data", [])]
            print(f"{OK}  vLLM is running — models: {models}")
            return True
    except urllib.error.URLError as e:
        print(f"{FAIL} Cannot reach {base_url}: {e.reason}")
        print("     → Start vLLM: vllm serve qwen2.5-14b --port 8000  (or check your job)")
        return False
    except Exception as e:
        print(f"{FAIL} vLLM check failed: {e}")
        return False


def check_llm_only():
    print("\n[4] V0 LLM-only end-to-end call")
    try:
        from src.utils.config_loader import load_config
        from src.pipeline.llm_only import LLMOnlyPipeline
        cfg = load_config()
        pipeline = LLMOnlyPipeline(cfg)
        t0 = time.time()
        result = pipeline.ask("What is Microsoft's primary cloud platform?")
        latency = time.time() - t0
        answer = result.get("answer", "")
        if result.get("error"):
            print(f"{FAIL} Pipeline returned error: {result['error']}")
            return False
        print(f"{OK}  Answer ({latency:.1f}s): {answer[:120]}...")
        return True
    except Exception as e:
        print(f"{FAIL} LLM-only call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_v1_retrieval():
    print("\n[5] V1 baseline retrieval call (dense only)")
    try:
        from src.utils.config_loader import load_config
        from src.pipeline.baseline import BaselinePipeline
        cfg = load_config()
        pipeline = BaselinePipeline(cfg)
        t0 = time.time()
        result = pipeline.ask("What was Microsoft's total revenue in FY2023?")
        latency = time.time() - t0
        n_chunks = len(result.get("retrieved_chunks", []))
        answer = result.get("answer", "")
        if result.get("error"):
            print(f"{FAIL} Pipeline returned error: {result['error']}")
            return False
        print(f"{OK}  Retrieved {n_chunks} chunks, answered in {latency:.1f}s")
        print(f"     Answer: {answer[:120]}...")
        return True
    except Exception as e:
        print(f"{FAIL} V1 retrieval call failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  FinSight Cluster Diagnostic")
    print("=" * 60)

    results = {
        "chroma":     check_chroma(),
        "bm25":       check_bm25(),
        "vllm":       check_vllm(),
        "llm_only":   False,
        "v1_retrieval": False,
    }

    # Only run LLM-only if vLLM is up
    if results["vllm"]:
        results["llm_only"] = check_llm_only()
    else:
        print("\n[4] V0 LLM-only call  — SKIPPED (vLLM not reachable)")

    # Only run V1 if both indexes and vLLM are up
    if results["chroma"] and results["bm25"] and results["vllm"]:
        results["v1_retrieval"] = check_v1_retrieval()
    else:
        print("\n[5] V1 retrieval call — SKIPPED (indexes or vLLM not ready)")

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    labels = {
        "chroma":       "ChromaDB index",
        "bm25":         "BM25 index",
        "vllm":         "vLLM endpoint",
        "llm_only":     "V0 LLM-only call",
        "v1_retrieval": "V1 retrieval call",
    }
    all_ok = True
    for key, label in labels.items():
        status = OK if results[key] else FAIL
        print(f"{status}  {label}")
        if not results[key]:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("  All checks passed — ready to run evaluation!")
        print("  python evaluation/run_evaluation.py --limit 3 --skip-ragas")
    else:
        print("  Fix the failing checks above, then re-run this script.")
    print()


if __name__ == "__main__":
    main()
