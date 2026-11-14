"""Write LaTeX tables for the paper from results/*.json (no hand-copied numbers).

python src/tables.py  -> paper/tables/{main,ablate}.tex
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RES / f"{name}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if "scenarios" in d else None


def scen_means(d):
    return {r["scenario"]: -r["mean"] for r in d["scenarios"]}


def climate_groups(d):
    cold, mild, hot = [], [], []
    for r in d["scenarios"]:
        c = r["scenario"].split("/")[1].split("_")[0]
        (cold if c in ("fairbanks", "minneapolis", "denver") else hot if c in ("phoenix", "singapore") else mild).append(-r["mean"])
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return m(cold), m(mild), m(hot)


def fmt(x):
    return f"{x * 1000:.3f}"


def main_table(rows):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Controller & Scenarios, simulator & Score & Comfort & Energy & Carbon & Cold / Mild / Hot \\", r"\midrule"]
    for row in rows:
        if row is None:
            lines.append(r"\midrule")
            continue
        label, name, sim = row
        d = load(name)
        if d is None:
            lines.append(f"{label} & {sim} & \\TBD{{pending}} & & & & \\\\")
            continue
        s = d["summary"]
        c, m, h = climate_groups(d)
        lines.append((r"\rowcolor[HTML]{EDF3FC}" if "Ours" in label else "") + f"{label} & {sim} & {fmt(s['score'])} & {fmt(s['comfort'])} & {fmt(s['energy'])} & "
                     f"{fmt(s['carbon'])} & {fmt(c)} / {fmt(m)} / {fmt(h)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def paired(ref, name):
    a, b = scen_means(load(ref)), scen_means(load(name))
    assert set(a) == set(b), f"{name}: scenario set differs from {ref}"
    sa, sb = sum(a.values()) / len(a), sum(b.values()) / len(b)
    wins = sum(b[k] < a[k] - 1e-9 for k in a)
    ties = sum(abs(b[k] - a[k]) <= 1e-9 for k in a)
    return sb, 100 * (sb - sa) / sa, wins, ties


def ablate_table(sections):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Variant (10-day val16) & Score & $\Delta$ vs.\ ref. & Better / 16 \\"]
    for title, ref, rows in sections:
        r0 = load(ref)
        if r0 is None:
            lines += [r"\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}} \\TBD{{pending}}}} \\\\"]
            continue
        assert all(r["steps"] == r0["scenarios"][0]["steps"] for r in r0["scenarios"])
        base = sum(scen_means(r0).values()) / len(r0["scenarios"])
        lines += [r"\midrule", f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}} (ref.\\ {fmt(base)})}} \\\\"]
        for label, name in rows:
            if load(name) is None:
                lines.append(f"{label} & \\TBD{{pending}} & & \\\\")
                continue
            sb, pct, wins, ties = paired(ref, name)
            w = f"{wins}" + (f" ({ties} tied)" if ties else "")
            lines.append((r"\rowcolor[HTML]{EDF3FC}" if "Ours" in label else "") + f"{label} & {fmt(sb)} & " + f"{pct:+.2f}".replace("-", "$-$") + f"\\% & {w} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


MAIN = [
    ("PID + water reset (organizer baseline)", "pidw_val_fixed", "val16, corrected"),
    ("Public chain~\\citep{penguin2026}", "penguin_val_fixed", "val16, corrected"),
    ("Public chain, $\\lambda_{\\mathrm{gas}}=0.5$ (v1)", "g50_val_fixed", "val16, corrected"),
    ("Ours, 72 h hold (v2b)", "v2b_val_fixed", "val16, corrected"),
    ("\\textbf{Ours} (v2a)", "v2a_val_fixed", "val16, corrected"),
    None,
    ("PID + water reset (organizer baseline)", "pidw_val_shipped", "val16, shipped"),
    ("Public chain~\\citep{penguin2026}", "penguin_val_shipped", "val16, shipped"),
    ("Ours, 72 h hold (v2b)", "v2b_val_shipped", "val16, shipped"),
    ("\\textbf{Ours} (v2a)", "v2a_val_shipped", "val16, shipped"),
    None,
    ("Public chain~\\citep{penguin2026}", "penguin_train30_fixed", "public 30, corrected"),
    ("\\textbf{Ours} (v2a)", "v2a_train30_fixed", "public 30, corrected"),
]
ABLATE = [
    ("Corrected simulator: cost weight, ref.\\ public chain", "s10_penguin", [
        ("Gas price $\\lambda_{\\mathrm{gas}}$ 0.30 $\\to$ 0.50", "s10_mine"),
    ]),
    ("Corrected simulator: our changes, ref.\\ public chain with $\\lambda_{\\mathrm{gas}}=0.5$", "s10_mine", [
        ("Comfort margin 0.08 $\\to$ 0.04 K", "s10_mine_m04"),
        ("Comfort margin 0.08 $\\to$ 0.16 K", "s10_mine_m16"),
        ("Standby floor 274 $\\to$ 280 K (frost limit)", "s10_mine_sb280"),
        ("Standby ceiling 307 $\\to$ 303 K", "s10_mine_smax303"),
        ("Standby ceiling 307 $\\to$ 300 K", "s10_mine_smax300"),
        ("Pre-conditioning ramp 1 $\\to$ 0.5 h", "s10_mine_pre05"),
        ("Water hold after demand 72 $\\to$ 12 h", "s10_hold12"),
        ("Water hold after demand 72 $\\to$ 2 h", "s10_hold2"),
        ("Water reservoir floor 320 $\\to$ 310 K", "s10_floor310"),
        ("Water reservoir floor 320 $\\to$ 335 K", "s10_floor335"),
        ("Opt.\\ start $\\ell_0$ 0.5 h, $\\ell_1$ 0.35 h/K, cap 10 h", "s10_opt"),
        ("Opt.\\ start $\\ell_0$ 0.25 h, $\\ell_1$ 0.10 h/K, cap 3 h", "s10_opt_k01"),
        ("Opt.\\ start $\\ell_0$ 0.25 h, $\\ell_1$ 0.07 h/K, cap 3 h", "s10_opt_k007"),
        ("Opt.\\ start $\\ell_0$ 0.25 h, $\\ell_1$ 0.05 h/K, cap 3 h", "s10_opt_k005"),
        ("Opt.\\ start $\\ell_0$ 0.25 h, $\\ell_1$ 0.03 h/K, cap 3 h", "s10_opt_k003"),
        ("Opt.\\ start, 0.05 h/K, only $>$1 h before comfort", "s10_opt_early"),
        ("Ceiling 303 K + 12 h hold", "s10_combo1"),
        ("Ceiling 303 K + 2 h hold", "s10_combo2"),
        ("\\textbf{Ours} (v2a): opt.\\ start 0.05 h/K + ceiling 303 K + 2 h hold", "s10_v2a"),
        ("Ours, keeping the 72 h hold (v2b)", "s10_v2b"),
        ("Oracle: per-zone departures (relax empty zones after 13:00)", "s10_or_dep"),
    ]),
    ("Corrected simulator: occupancy oracle, ref.\\ public chain", "s10_penguin", [
        ("Schedule-only control with the oracle's band formula", "s10_sched"),
        ("Oracle: future per-zone occupancy, same band formula", "s10_or_o2"),
    ]),
    ("Shipped simulator, ref.\\ public chain with $\\lambda_{\\mathrm{gas}}=0.5$", "s10sh_mine", [
        ("Water hold after demand 72 $\\to$ 2 h", "s10sh_hold2"),
        ("Ours, keeping the 72 h hold (v2b)", "s10sh_v2b"),
        ("\\textbf{Ours} (v2a)", "s10sh_v2a"),
    ]),
]

ABLATE_CHAIN = [
    ("Corrected simulator: public chain components, ref.\\ public chain", "s10_penguin", [
        ("Base MPC only~\\citep{wu2026cpu}", "s10_wu_base"),
        ("+ error reserve", "s10_v42"),
        ("+ identified response gain, 36 h warm-water standby", "s10_c_warm"),
        ("+ 72 h hold, 274 K setback, fast recovery", "s10_c_deep"),
        ("+ empty-day relaxation, 0.5 K SAT grid, error-scaled margin", "s10_c_stable"),
        ("+ demand-sized water temperature (320 K floor)", "s10_c_dem"),
    ]),
]


def val_table():
    d = load("penguin_val_fixed")
    by = {}
    for r in d["scenarios"]:
        b, w = r["scenario"].split("/")
        by.setdefault(b, [r["n_ahu"], r["n_zone"], []])[2].append(w.replace("_", r"\_"))
    lines = [r"\begin{tabular}{lrrl}", r"\toprule", r"Building & AHUs & Zones & Weather strips \\", r"\midrule"]
    for b, (a, z, ws) in sorted(by.items()):
        lines.append(f"{b.replace('_', r'\_')} & {a} & {z} & {', '.join(sorted(ws))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def exploit_table():
    p = RES / "exploit_aggregates.json"
    agg = json.loads(p.read_text())
    labels = [("base", "Public chain"), ("saw", "+ tank sawtooth"), ("o2", "+ future occupancy (oracle)"),
              ("sawo2", "+ sawtooth and future occupancy")]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Shipped simulator} & \multicolumn{2}{c}{Corrected simulator} \\",
             r"Policy & Score & Gain & Score & Gain \\", r"\midrule"]
    for key, lab in labels:
        cells = []
        for sim in ("starter-kit", "kit-fixed"):
            a = agg.get(f"{sim}/{key}")
            if not a or a.get("n", 0) < 5:
                cells += [r"\TBD{pending}", ""]
            else:
                cells += [fmt(a["score"]), "--" if key == "base" else f"{100 * a['gain']:+.1f}".replace("-", "$-$") + "\\%"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    (OUT / "exploit.tex").write_text(exploit_table())
    (OUT / "val16.tex").write_text(val_table())
    (OUT / "main.tex").write_text(main_table(MAIN))
    (OUT / "ablate.tex").write_text(ablate_table(ABLATE))
    (OUT / "ablate_chain.tex").write_text(ablate_table(ABLATE_CHAIN))
    print((OUT / "main.tex").read_text())
    print((OUT / "ablate.tex").read_text())
