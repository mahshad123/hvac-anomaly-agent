"""CLI.

    python -m hvac_agent run   [--agent rule|claude] [--ship S1] [--seed 7]   # writes report
    python -m hvac_agent eval  [--agent rule|claude] [--seeds 7 11 23]         # benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .agent import ClaudeAgent, RuleBasedAgent
from .detect import naive_threshold_flags
from .evaluate import score, score_naive
from .report import render_report
from .simulate import FleetConfig, simulate_fleet
from .tools import Toolbox


def _agent(name: str):
    return ClaudeAgent() if name == "claude" else RuleBasedAgent()


def cmd_run(args) -> None:
    tel, truth = simulate_fleet(FleetConfig(seed=args.seed))
    tb = Toolbox(tel)
    agent = _agent(args.agent)
    diags = agent.run(tb, ship_id=args.ship)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(diags))
    (out / "diagnoses.json").write_text(json.dumps([d.to_dict() for d in diags], indent=2, default=str))
    print(render_report(diags))
    print(f"[{len(tb.calls)} tool calls] wrote {out / 'report.md'}")


def cmd_eval(args) -> None:
    rows = []
    for seed in args.seeds:
        tel, truth = simulate_fleet(FleetConfig(seed=seed))
        rows.append({"method": "naive_threshold", "seed": seed, **score_naive(naive_threshold_flags(tel), truth)})
        tb = Toolbox(tel)
        # Ablation: the statistical detector alone, paging the crew for every flag.
        rows.append({"method": "detector_only", "seed": seed, **score_naive(tb.signals["flagged"], truth)})
        diags = _agent(args.agent).run(tb)
        rows.append({"method": f"agent_{args.agent}", "seed": seed, **score(diags, truth)})
    df = pd.DataFrame(rows)
    agg = df.drop(columns="seed").groupby("method", sort=False).mean().round(3)
    print(agg.T.to_markdown())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"eval_{args.agent}.md").write_text(agg.T.to_markdown() + "\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="hvac_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", choices=["rule", "claude"], default="rule")
    r.add_argument("--ship", default=None)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--out", default="results")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("eval")
    e.add_argument("--agent", choices=["rule", "claude"], default="rule")
    e.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23, 42, 99])
    e.add_argument("--out", default="results")
    e.set_defaults(fn=cmd_eval)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
