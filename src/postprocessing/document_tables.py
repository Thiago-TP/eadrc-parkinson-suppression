"""
Updates the numbers of the results tables of the CBA 2026 paper and presentation
from the post-processed results, keeping each table's layout.

- Average metrics (mean +- standard deviation over the 100 runs), from
  results/metrics/<strategy>_amplitude_<amplitude>_metrics-summary.csv
- Wilcoxon tests (Holm-Bonferroni corrected p and rank-biserial r), from
  results/stats/wilcoxon_full_results.csv

Only the cells are rewritten: the paper tables use decimal points, the slide
tables decimal commas, $-$ for minus signs and \\alert on non-significant tests.

Run from the repository root, after src/postprocessing/postprocess.py:
    uv run src/postprocessing/document_tables.py
"""

import re
from collections.abc import Callable

import pandas as pd

STRATEGIES = {
    "EADRC+EBMFLC": "eadrc_ebmflc",
    "EADRC+ZPLP": "eadrc_zplp",
    "DE-PID": "pid_de",
    "IMC-PID": "pid_imc",
}
COLUMNS = ["tpsr_percent", "r2", "response_entropy", "iae", "control_power", "control_tvc"]
AMPLITUDES = ["0.0", "1.0"]  # order of the scenarios in every table: rest, then motion
SCENARIOS = ["Resting (tau_v=0)", "Deliberate motion (tau_v given by eq.22)"]
BASELINES = ["EADRC+EBMFLC", "DE-PID", "IMC-PID"]
ALPHA = 0.05

PAPER = "cba-2026-paper/tables"
SLIDES = "cba-2026-presentation/tables"

# Metric row labels of each Wilcoxon table, and the metric they hold
PAPER_METRICS = {
    r"TPSR [\%]": "TPSR [%]",
    r"$R^2$": "R2",
    "Entropy [bits]": "Entropy [bits]",
    r"IAE [rad$\times$s]": "IAE [rad x s]",
    r"Power [(N$\times$m)$^2$]": "Power [(Nm)^2]",
    r"TV [N$\times$m$\times$s]": "TV [Nm x s]",
}
SLIDE_METRICS = {
    "TPSR": "TPSR [%]",
    r"$R^2$": "R2",
    "Entropia": "Entropy [bits]",
    "IAE": "IAE [rad x s]",
    "Potência": "Power [(Nm)^2]",
    "TV": "TV [Nm x s]",
}


def paper_number(x: float, decimals: int = 2, sign: bool = False) -> str:
    return f"{x:+.{decimals}f}" if sign else f"{x:.{decimals}f}"


def slide_number(x: float, decimals: int = 2, sign: bool = False) -> str:
    s = paper_number(x, decimals, sign).replace(".", ",")
    return "$-$" + s[1:] if s.startswith("-") else s


def summary(strategy: str, amplitude: str) -> pd.DataFrame:
    path = f"results/metrics/{STRATEGIES[strategy]}_amplitude_{amplitude}_metrics-summary.csv"
    return pd.read_csv(path).set_index("statistic")


def update_average_table(path: str, row: re.Pattern, cell: re.Pattern, fmt: Callable[[float, float], str]) -> None:
    lines = open(path).read().split("\n")
    seen: dict[str, int] = {}
    for i, line in enumerate(lines):
        match = row.match(line)
        if not match:
            continue
        strategy = match.group(1)
        scenario = seen.get(strategy, 0)
        seen[strategy] = scenario + 1
        stats = summary(strategy, AMPLITUDES[scenario])
        cells = iter(fmt(stats.loc["mean", c], stats.loc["standard deviation", c]) for c in COLUMNS)
        lines[i], count = cell.subn(lambda _: next(cells), line)
        assert count == len(COLUMNS), f"{path}: {count} cells in row {line!r}"
    assert seen == {s: len(AMPLITUDES) for s in STRATEGIES}, f"{path}: rows found {seen}"
    open(path, "w").write("\n".join(lines))


def update_wilcoxon_table(path: str, labels: dict[str, str], fmt: Callable[[pd.Series], list[str]]) -> None:
    results = pd.read_csv("results/stats/wilcoxon_full_results.csv")
    lines = open(path).read().split("\n")
    seen: dict[str, int] = {}
    for i, line in enumerate(lines):
        for label, metric in labels.items():
            match = re.match(rf"^(.*?&\s*{re.escape(label)}\s*&\s*)", line)
            if not match:
                continue
            scenario = seen.get(label, 0)
            seen[label] = scenario + 1
            cells = []
            for baseline in BASELINES:
                test = results[
                    (results.scenario == SCENARIOS[scenario])
                    & (results.metric == metric)
                    & (results.comparison == f"EADRC+ZPLP vs {baseline}")
                ].iloc[0]
                cells += fmt(test)
            lines[i] = match.group(1) + " & ".join(cells) + " \\\\"
    assert seen == {label: len(SCENARIOS) for label in labels}, f"{path}: rows found {seen}"
    open(path, "w").write("\n".join(lines))


def paper_test(test: pd.Series) -> list[str]:
    p = "$<$0.001" if test.p_holm < 0.001 else paper_number(test.p_holm, 3)
    return [p, paper_number(test.rank_biserial_r, sign=True)]


def slide_test(test: pd.Series) -> list[str]:
    p = "$<$0,001" if test.p_holm < 0.001 else slide_number(test.p_holm, 3)
    r = slide_number(test.rank_biserial_r, sign=True)
    if test.p_holm >= ALPHA:
        p, r = f"\\alert{{{p}}}", f"\\alert{{{r}}}"
    return [p, r]


if __name__ == "__main__":
    strategies = "|".join(re.escape(s) for s in STRATEGIES)
    update_average_table(
        f"{PAPER}/table_average_metrics.tex",
        row=re.compile(rf"^({strategies}) &"),  # commented-out rows start with %
        cell=re.compile(r"-?\d+\.\d+ \$\\pm\$ \d+\.\d+"),
        fmt=lambda m, s: f"{paper_number(m)} $\\pm$ {paper_number(s)}",
    )
    update_average_table(
        f"{SLIDES}/table_average_metrics.tex",
        row=re.compile(rf"^\s*& ({strategies}) &"),
        cell=re.compile(r"\\meanstd\{[^}]*\}\{[^}]*\}"),
        fmt=lambda m, s: f"\\meanstd{{{slide_number(m)}}}{{{slide_number(s)}}}",
    )
    update_wilcoxon_table(f"{PAPER}/table_wilcoxon.tex", PAPER_METRICS, paper_test)
    update_wilcoxon_table(f"{SLIDES}/table_wilcoxon.tex", SLIDE_METRICS, slide_test)
    print("Updated the average metrics and Wilcoxon tables of the paper and the presentation.")
