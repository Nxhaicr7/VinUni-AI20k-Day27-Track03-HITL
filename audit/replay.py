"""Replay a review session from the audit trail.

Usage:
    uv run python -m audit.replay --thread <thread_id>
    uv run python -m audit.replay --list                # list recent threads

The script reads `audit_events` (human-readable timeline). The LangGraph
SqliteSaver checkpoint tables live in the same .db file but are queried
separately by the LangGraph runtime, not by this tool.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from common.db import db_conn, replay_events


RISK_COLOR = {"low": "green", "med": "yellow", "high": "red"}


async def list_threads() -> None:
    console = Console()
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()

    table = Table(title="Recent review sessions")
    for col in ("thread_id", "pr_url", "started", "last_event", "worst_risk", "events"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r["thread_id"],
            r["pr_url"],
            str(r["started"]),
            str(r["last_event"]),
            str(r["worst_risk"]),
            str(r["events"]),
        )
    console.print(table)


async def replay(thread_id: str) -> None:
    console = Console()
    events = await replay_events(thread_id)

    if not events:
        console.print(f"[red]No events found for thread {thread_id}[/red]")
        return

    console.rule(f"[bold]Replay {thread_id}")
    for ev in events:
        risk = ev["risk_level"]
        risk_colored = f"[{RISK_COLOR.get(risk, 'white')}]{risk:<4}[/]"
        reviewer = ev["reviewer_id"] or "-"
        reason = (ev["reason"] or "")[:60]
        console.print(
            f"[dim]{ev['timestamp']}[/dim]  "
            f"[cyan]{ev['action']:<18}[/cyan] "
            f"conf=[bold]{ev['confidence']:.2f}[/bold] "
            f"risk={risk_colored} "
            f"decision=[magenta]{ev['decision']:<10}[/magenta] "
            f"reviewer={reviewer:<14} "
            f"{ev['execution_time_ms']:>5}ms  "
            f"[dim]{reason}[/dim]"
        )


async def calibrate_confidence() -> None:
    """Bonus 2: Compute confidence calibration stats from audit_events.
    
    Shows AVG(confidence) for approved reviews vs overall approval rate.
    Helps determine if the model is over- or under-confident.
    """
    console = Console()
    async with db_conn() as conn:
        # Get stats for human-approved reviews
        async with conn.execute(
            """
            SELECT 
                COUNT(*) as total_approvals,
                AVG(confidence) as avg_confidence_approved,
                MIN(confidence) as min_confidence_approved,
                MAX(confidence) as max_confidence_approved
            FROM audit_events 
            WHERE action = 'human_approval' AND decision = 'approve'
            """
        ) as cur:
            approved_stats = await cur.fetchone()
        
        # Get total human reviews
        async with conn.execute(
            """
            SELECT COUNT(*) as total_human_reviews
            FROM audit_events 
            WHERE action = 'human_approval'
            """
        ) as cur:
            total_human = await cur.fetchone()
        
        # Get overall stats
        async with conn.execute(
            """
            SELECT 
                AVG(confidence) as overall_avg_confidence,
                COUNT(*) as total_sessions
            FROM audit_events 
            WHERE action = 'analyze'
            """
        ) as cur:
            overall_stats = await cur.fetchone()

    if not approved_stats or not total_human or not overall_stats:
        console.print("[red]No data available for calibration.[/red]")
        return

    total_approvals = approved_stats["total_approvals"] or 0
    total_human_reviews = total_human["total_human_reviews"] or 0
    approval_rate = total_approvals / total_human_reviews if total_human_reviews > 0 else 0
    
    avg_conf_approved = approved_stats["avg_confidence_approved"] or 0
    overall_avg_conf = overall_stats["overall_avg_confidence"] or 0
    
    console.print("[bold]Confidence Calibration Report[/bold]")
    console.print(f"Total sessions analyzed: {overall_stats['total_sessions']}")
    console.print(f"Total human reviews: {total_human_reviews}")
    console.print(f"Approval rate: {approval_rate:.1%}")
    console.print()
    console.print(f"Average confidence (all reviews): {overall_avg_conf:.2%}")
    console.print(f"Average confidence (approved only): {avg_conf_approved:.2%}")
    console.print()
    
    if avg_conf_approved > approval_rate:
        console.print("[yellow]⚠️  Model appears OVER-CONFIDENT[/yellow]")
        console.print("   (Average confidence of approved reviews > actual approval rate)")
    elif avg_conf_approved < approval_rate:
        console.print("[yellow]⚠️  Model appears UNDER-CONFIDENT[/yellow]")
        console.print("   (Average confidence of approved reviews < actual approval rate)")
    else:
        console.print("[green]✅ Model confidence well-calibrated[/green]")
    
    console.print()
    console.print("Approved reviews confidence range:")
    if approved_stats["min_confidence_approved"] is not None:
        console.print(f"  Min: {approved_stats['min_confidence_approved']:.1%}")
        console.print(f"  Max: {approved_stats['max_confidence_approved']:.1%}")
    else:
        console.print("  No approved reviews yet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", help="Replay one thread by id")
    parser.add_argument("--list", action="store_true", help="List recent threads")
    parser.add_argument("--calibrate", action="store_true", help="Show confidence calibration stats")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_threads())
    elif args.thread:
        asyncio.run(replay(args.thread))
    elif args.calibrate:
        asyncio.run(calibrate_confidence())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
