"""
Report generation.

Turns an EvalSummary into a human-readable markdown report or a standalone HTML
report. The HTML is self-contained (inline CSS, no external assets) so it can be
opened directly or committed as an artifact.
"""

import html
from typing import List

from .types import EvalSummary


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x == x else "n/a"


def to_markdown(summary: EvalSummary) -> str:
    lines: List[str] = []
    lines.append("# RAG evaluation report")
    lines.append("")
    lines.append(f"- adapter: `{summary.config.get('adapter', '?')}`")
    lines.append(f"- judge: `{summary.config.get('judge', '?')}`")
    lines.append(f"- cases: {summary.n_cases}")
    lines.append(f"- latency p50: {summary.latency_p50_ms:.1f}ms, "
                 f"p95: {summary.latency_p95_ms:.1f}ms")
    if summary.total_prompt_tokens or summary.total_completion_tokens:
        lines.append(f"- tokens: {summary.total_prompt_tokens} prompt, "
                     f"{summary.total_completion_tokens} completion")
    lines.append("")
    lines.append("## metrics")
    lines.append("")
    lines.append("| metric | mean | std |")
    lines.append("|--------|------|-----|")
    for metric in summary.metric_means:
        mean = summary.metric_means[metric]
        std = summary.metric_stds.get(metric, 0.0)
        lines.append(f"| {metric} | {_fmt(mean)} | {_fmt(std)} |")
    lines.append("")

    lines.append("## per-case results")
    lines.append("")
    for i, r in enumerate(summary.results, 1):
        lines.append(f"### case {i}")
        lines.append(f"> {r.test_case.question}")
        lines.append("")
        lines.append(f"**answer:** {r.output.answer[:400]}")
        lines.append("")
        score_str = ", ".join(f"{m}={_fmt(v)}" for m, v in r.scores.items())
        lines.append(f"scores: {score_str}")
        if r.test_case.relevant_ids:
            lines.append(f"relevant ids: {r.test_case.relevant_ids}")
            lines.append(f"retrieved ids: {r.output.retrieved_ids}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG evaluation report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 920px; margin: 40px auto; padding: 0 20px; color: #1a1a2e;
         line-height: 1.5; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  .meta {{ color: #555; font-size: 14px; margin-bottom: 24px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: 12px; margin: 24px 0; }}
  .card {{ border: 1px solid #e3e3ef; border-radius: 10px; padding: 16px;
          background: #fafaff; }}
  .card .name {{ font-size: 13px; color: #666; text-transform: lowercase; }}
  .card .value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .card .bar {{ height: 6px; border-radius: 3px; margin-top: 8px;
               background: linear-gradient(90deg, #6c7ce0 var(--pct), #e3e3ef var(--pct)); }}
  .case {{ border: 1px solid #e3e3ef; border-radius: 10px; padding: 16px;
          margin: 14px 0; }}
  .case .q {{ font-weight: 600; }}
  .case .a {{ color: #333; margin: 8px 0; }}
  .scores {{ font-size: 13px; color: #555; }}
  .ids {{ font-size: 12px; color: #888; font-family: monospace; margin-top: 6px; }}
  code {{ background: #eef; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>RAG evaluation report</h1>
<div class="meta">
  adapter <code>{adapter}</code> &middot; judge <code>{judge}</code> &middot;
  {n_cases} cases &middot; latency p50 {p50:.1f}ms / p95 {p95:.1f}ms
</div>
<div class="metrics">
{metric_cards}
</div>
<h2>per-case results</h2>
{cases}
</body>
</html>"""


def to_html(summary: EvalSummary) -> str:
    cards = []
    for metric, mean in summary.metric_means.items():
        pct = int(mean * 100) if mean == mean else 0
        cards.append(
            f'<div class="card"><div class="name">{html.escape(metric)}</div>'
            f'<div class="value">{_fmt(mean)}</div>'
            f'<div class="bar" style="--pct:{pct}%"></div></div>')

    case_blocks = []
    for i, r in enumerate(summary.results, 1):
        score_str = ", ".join(f"{m}={_fmt(v)}" for m, v in r.scores.items())
        ids_html = ""
        if r.test_case.relevant_ids:
            ids_html = (
                f'<div class="ids">relevant: {html.escape(str(r.test_case.relevant_ids))}'
                f'<br>retrieved: {html.escape(str(r.output.retrieved_ids))}</div>')
        case_blocks.append(
            f'<div class="case"><div class="q">Q{i}. '
            f'{html.escape(r.test_case.question)}</div>'
            f'<div class="a">{html.escape(r.output.answer[:400])}</div>'
            f'<div class="scores">{html.escape(score_str)}</div>'
            f'{ids_html}</div>')

    return HTML_TEMPLATE.format(
        adapter=html.escape(summary.config.get("adapter", "?")),
        judge=html.escape(summary.config.get("judge", "?")),
        n_cases=summary.n_cases,
        p50=summary.latency_p50_ms,
        p95=summary.latency_p95_ms,
        metric_cards="\n".join(cards),
        cases="\n".join(case_blocks),
    )


def save_report(summary: EvalSummary, path: str) -> None:
    if path.endswith(".html"):
        content = to_html(summary)
    else:
        content = to_markdown(summary)
    with open(path, "w") as f:
        f.write(content)
