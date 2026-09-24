"""Autodiff puzzles: hunt sabotaged gradients with BackLens's X-ray tools.

Five puzzles, one corrupted gradient each. This demo solves puzzle p1 the
way a detective would -- then leaves p2-p5 for you.

Run:  python examples/08_puzzles.py
"""

import numpy as np

from backlens.puzzles import list_puzzles, load_puzzle
from backlens.debug import debug_backward, step_backward


def solve_like_a_detective(pid="p1"):
    """The method: diff tells you WHO is wrong; step-by-step comparison of
    clean vs sabotaged traces tells you WHERE the corruption enters."""
    p = load_puzzle(pid)
    print("=" * 68)
    print(f"{p.id}: {p.title}  [{p.difficulty}]  ({p.n_steps} backward steps)")
    print("=" * 68)
    print(p.story, "\n")

    # 1. WHO: which parameter gradients end up wrong?
    print("step 1 -- p.diff(): which parameters are damaged?")
    for line in p.diff().splitlines():
        print("   ", line)

    # 2. WHERE: walk both graphs (clean + sabotaged) side by side and find
    #    the first step whose gradient norms diverge
    live_loss, _, _ = p._build(False)
    clean_loss, _, _ = p._build(True)
    dirty_trace = debug_backward(live_loss, raise_on_anomaly=False)
    clean_trace = debug_backward(clean_loss, raise_on_anomaly=False)

    print("\nstep 2 -- first divergence between clean and sabotaged traces:")
    answer = None
    for d, c in zip(dirty_trace, clean_trace):
        marker = ""
        if not np.isclose(d.grad_norm, c.grad_norm, rtol=1e-3):
            answer = d.index
            marker = "   <== GRADIENT NORM DIVERGES HERE"
        print(f"   #{d.index:<3} {d.op:<12} |g| dirty={d.grad_norm:<12.4g} "
              f"clean={c.grad_norm:<12.4g}{marker}")
        if answer is not None:
            break

    # 3. verify with the puzzle's own referee
    print(f"\nstep 3 -- accusing step #{answer}: ", end="")
    ok = p.check(answer)
    print("CORRECT" if ok else "wrong (keep hunting)")
    return ok


if __name__ == "__main__":
    print(list_puzzles(), "\n")
    solve_like_a_detective("p1")
    print("\nyour turn: load_puzzle('p2') ... load_puzzle('p5')")
    print("tools: debug_backward(loss), step_backward(loss), p.diff(), p.check(i)")
