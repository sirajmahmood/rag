"""
run_all_demos.py
----------------
Convenience script: runs every stage's built-in demo in order, so you can see
each pipeline produce answers without invoking seven files by hand.

    python run_all_demos.py

For the scored side-by-side comparison instead, use:

    python eval/harness.py
"""

import importlib

STAGES = [
    "stages.stage_1_naive",
    "stages.stage_2_chunking",
    "stages.stage_3_hybrid",
    "stages.stage_4_rerank",
    "stages.stage_5_query",
    "stages.stage_6_contextual",
    "stages.stage_7_agentic",
]

if __name__ == "__main__":
    for name in STAGES:
        print("\n" + "#" * 78)
        print("#  " + name)
        print("#" * 78)
        mod = importlib.import_module(name)
        mod._demo()
    print("\nDone. For scored metrics across all stages, run: python eval/harness.py")
