"""Run-window guard: weekday experiments may only START at or after 5 PM America/New_York.

Weekends are unrestricted. `allowed(now)` is pure (inject `now` in tests); `require_window()` raises.
There is intentionally no environment override: change guard.weekday_start_hour in the config deliberately.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


class OutsideRunWindow(RuntimeError):
    pass


def now_in(tz: str = "America/New_York") -> datetime:
    return datetime.now(ZoneInfo(tz))


def allowed(now: datetime | None = None, tz: str = "America/New_York", weekday_start_hour: int = 17) -> tuple[bool, str]:
    now = (now or now_in(tz)).astimezone(ZoneInfo(tz))
    stamp = now.strftime("%a %Y-%m-%d %H:%M %Z")
    if now.weekday() >= 5:
        return True, f"weekend ({stamp}): runs allowed"
    if now.hour >= weekday_start_hour:
        return True, f"weekday after {weekday_start_hour}:00 ({stamp}): runs allowed"
    return False, (f"refusing to start: it is {stamp}; weekday runs may only start at or after "
                   f"{weekday_start_hour}:00 {tz}")


def require_window(cfg: dict | None = None, now: datetime | None = None) -> str:
    g = (cfg or {}).get("guard", {})
    ok, why = allowed(now, g.get("timezone", "America/New_York"), int(g.get("weekday_start_hour", 17)))
    if not ok:
        raise OutsideRunWindow(why)
    return why
