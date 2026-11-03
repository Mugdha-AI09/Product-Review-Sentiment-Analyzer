# ============================================================
# 🚀  run_all.py  —  Run the full pipeline end-to-end
#
#     Usage:
#       python run_all.py           → run ALL stages (1 → 2 → 3 → 4)
#       python run_all.py 1         → preprocess only
#       python run_all.py 1 2       → preprocess + EDA
#       python run_all.py 3         → train only  (needs stage 1 done first)
#       python run_all.py 4         → predict only (needs stage 3 done first)
# ============================================================

import sys
import time

import pipeline_1_preprocess as p1
import pipeline_2_eda        as p2
import pipeline_3_train      as p3
import pipeline_4_predict    as p4

STAGES = {
    "1": ("🧹  Preprocess",   p1.run),
    "2": ("📊  EDA Plots",    p2.run),
    "3": ("🤖  Train Models", p3.run),
    "4": ("💬  Predict",      p4.run),
}


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(STAGES.keys())

    for s in requested:
        if s not in STAGES:
            print(f"  ❌  Unknown stage '{s}'. Valid: 1 2 3 4")
            sys.exit(1)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     🎯  SENTIMENT ANALYSIS PIPELINE              ║")
    stages_str = " → ".join(f"Stage {s}" for s in requested)
    print(f"║     Running: {stages_str:<37}║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    total_start = time.time()

    for s in requested:
        label, fn = STAGES[s]
        print(f"\n{'─'*56}")
        print(f"  STAGE {s}: {label}")
        print(f"{'─'*56}")
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"  ⏱️   Stage {s} completed in {elapsed:.1f}s")

    total = time.time() - total_start
    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  ✅  All done!  Total time: {total:.1f}s{' '*(22-len(f'{total:.1f}'))}║")
    print("╚══════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
