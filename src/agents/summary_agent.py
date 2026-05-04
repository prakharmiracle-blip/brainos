"""
BrainOS — Daily Summary Agent
Generates structured daily insights from recently ingested memories.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, date

from src.memory.vector_store import vector_store
from src.utils.llm_client import llm_client
from src.utils.logger import log
from src.utils.models import SummaryResponse

SUMMARY_SYSTEM = """You are BrainOS, a personal AI memory summariser.
Given a set of memory entries from today, produce a structured daily summary.

Respond ONLY with valid JSON in this exact format:
{
  "insights": ["insight 1", "insight 2", "insight 3"],
  "key_learnings": ["learning 1", "learning 2"],
  "important_notes": ["note 1", "note 2"]
}

Rules:
- insights: Top 3 actionable or interesting observations
- key_learnings: Concepts or knowledge the user captured
- important_notes: Reminders, tasks, or flags to revisit
- If there's nothing in a category, return an empty list []
- Be specific and use the actual content — don't be generic
- Maximum 3 items per category
"""


class SummaryAgent:

    def generate(self, n_recent: int = 100, target_date: date | None = None) -> SummaryResponse:
        """
        Summarise the most recent memories (default: last 100 chunks).
        Optionally filter to a specific date.
        """
        target_date = target_date or date.today()
        log.info(f"Generating daily summary for {target_date}")

        recent = vector_store.get_recent(n=n_recent)

        # Filter to target date if needed
        day_entries = []
        for entry in recent:
            ts = entry.get("timestamp", "")
            try:
                entry_date = datetime.fromisoformat(ts).date()
                if entry_date == target_date:
                    day_entries.append(entry)
            except Exception:
                day_entries.append(entry)  # include if timestamp unparseable

        if not day_entries:
            # Fall back to all recent if nothing for today
            log.info("No entries for today — summarising all recent entries.")
            day_entries = recent

        if not day_entries:
            return SummaryResponse(
                date=str(target_date),
                insights=["No memories captured yet."],
                key_learnings=[],
                important_notes=[],
                raw_summary="Nothing stored yet.",
            )

        # Build a condensed text block of all entries
        memory_block = self._format_entries(day_entries)

        user_prompt = f"""Date: {target_date}

Here are today's memory entries:
{memory_block}

Generate the JSON summary now."""

        raw = llm_client.chat(
            system=SUMMARY_SYSTEM,
            user=user_prompt,
            max_tokens=1024,
        )

        parsed = self._parse_json(raw)

        return SummaryResponse(
            date=str(target_date),
            insights=parsed.get("insights", []),
            key_learnings=parsed.get("key_learnings", []),
            important_notes=parsed.get("important_notes", []),
            raw_summary=raw,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _format_entries(self, entries: list[dict]) -> str:
        lines = []
        for i, e in enumerate(entries, 1):
            ts = e.get("timestamp", "")[:19]
            itype = e.get("input_type", "text")
            content = e.get("content", "")[:400]
            lines.append(f"[{i}] {ts} ({itype}): {content}")
        return "\n".join(lines)

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response robustly."""
        try:
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            return json.loads(clean)
        except json.JSONDecodeError:
            log.warning("Could not parse JSON from LLM summary response.")
            return {
                "insights": [raw[:300]],
                "key_learnings": [],
                "important_notes": [],
            }


# Singleton
summary_agent = SummaryAgent()
