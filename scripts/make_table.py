"""Collect evaluation JSONs into the ablation table (markdown + LaTeX rows)."""
import json
import os
import sys

ROWS = [("g_only", "G only (ignores the instruction)"),
        ("additive", "GR-ConvNet + CLIP, additive (LGD baseline)"),
        ("gxs_zs", "G x S, zero-shot S (temperature/bias only)"),
        ("gxs_full", "G x S, S loss on all patches"),
        ("gxs", "G x S (ours)")]


def main(d):
    res = {}
    for f in os.listdir(d):
        if f.endswith(".json"):
            tag, split = f[:-5].rsplit("_test_", 1)
            res[(tag, "test_" + split)] = json.load(open(os.path.join(d, f)))["summary"]
    pct = lambda v: f"{100 * v:.1f}" if v == v else "-"
    lines = ["| method | seen succ | seen succ (multi) | seen paired | seen select | unseen succ |",
             "|---|---|---|---|---|---|"]
    tex = []
    for tag, name in ROWS:
        s, u = res.get((tag, "test_seen")), res.get((tag, "test_unseen"))
        if not s:
            continue
        cells = [pct(s["success"]), pct(s["success_multi"]), pct(s["paired"]), pct(s["select"]),
                 pct(u["success"]) if u else "-"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        tex.append(f"{name} & " + " & ".join(cells) + r" \\")
    any_s = next(iter(res.values()))
    note = (f"n(seen)={res.get(('gxs','test_seen'), any_s)['n']}, "
            f"multi={res.get(('gxs','test_seen'), any_s)['n_multi']}, "
            f"pairs={res.get(('gxs','test_seen'), any_s)['n_pairs']}")
    out = "\n".join(lines) + "\n\n" + note + "\n"
    print(out)
    with open(os.path.join(d, "table.md"), "w") as f:
        f.write(out)
    with open(os.path.join(d, "table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
