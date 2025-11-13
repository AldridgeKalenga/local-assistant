# calendar_tools.py
# Google Calendar access and natural-language interpretation.

import os
import datetime
import re
from dateutil import tz, parser as dtparser

from config import (
    GCAL_SCOPES,
    GCAL_TOKEN_PATH,
    GCAL_CREDENTIALS_PATH,
    LOCAL_TZ_NAME,
)

# Try to import Google Calendar libs
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GCAL_AVAILABLE = True
except Exception:
    GCAL_AVAILABLE = False

# -------- Google Calendar low-level helpers --------

def gcal_build_service(identity="Aldridge"):
    """
    Build a Google Calendar service client for a specific persona's calendar.
    identity is used to pick token_<identity>.json and credentials_<identity>.json.
    """
    if not GCAL_AVAILABLE:
        raise RuntimeError("Google Calendar libs not installed.")

    token_path = f"token_{identity.lower()}.json"
    creds_path = f"credentials_{identity.lower()}.json"
    if not os.path.exists(creds_path):
        # fallback to shared creds path
        creds_path = GCAL_CREDENTIALS_PATH

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GCAL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # will open browser for auth on first run for that persona
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, GCAL_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)

def gcal_list_upcoming(
    n=10,
    time_min=None,
    time_max=None,
    tzname=LOCAL_TZ_NAME,
    identity="Aldridge"
):
    """
    List upcoming events within optional [time_min, time_max].
    Returns a list of event dicts.
    """
    service = gcal_build_service(identity=identity)
    local_tz = tz.gettz(tzname)
    now_local = datetime.datetime.now(local_tz)

    if time_min is None:
        time_min = now_local

    if isinstance(time_min, datetime.datetime):
        if time_min.tzinfo is None:
            time_min = time_min.replace(tzinfo=local_tz)
    else:
        time_min = dtparser.parse(str(time_min))
    time_min = time_min.astimezone(datetime.timezone.utc)

    params = {
        "calendarId": "primary",
        "timeMin": time_min.isoformat(),
        "singleEvents": True,
        "orderBy": "startTime",
        "maxResults": n
    }

    if time_max is not None:
        if isinstance(time_max, datetime.datetime):
            if time_max.tzinfo is None:
                time_max = time_max.replace(tzinfo=local_tz)
        else:
            time_max = dtparser.parse(str(time_max))
        time_max = time_max.astimezone(datetime.timezone.utc)
        params["timeMax"] = time_max.isoformat()

    events_result = service.events().list(**params).execute()
    return events_result.get("items", [])

def gcal_format_event(ev, tzname=LOCAL_TZ_NAME):
    """
    Format a calendar API event dict into a human-friendly line.
    """
    local = tz.gettz(tzname)
    start = ev["start"].get("dateTime") or ev["start"].get("date")
    end   = ev["end"].get("dateTime")   or ev["end"].get("date")

    def _fmt(dtstr):
        # all-day events come back as just YYYY-MM-DD
        if len(dtstr) <= 10:
            d = dtparser.parse(dtstr).date()
            return d.strftime("%a %b %d (all day)")
        dt = dtparser.parse(dtstr).astimezone(local)
        return dt.strftime("%a %b %d, %I:%M %p")

    title = ev.get("summary", "(no title)")
    loc = ev.get("location")
    line = f"- {title} | {_fmt(start)} → {_fmt(end)}"
    if loc:
        line += f" | {loc}"
    return line

def gcal_is_free_between(start_str, end_str, tzname=LOCAL_TZ_NAME, identity="Aldridge"):
    """
    Check free/busy for given local time window.
    Returns (is_free_bool, busy_list)
    """
    service = gcal_build_service(identity=identity)

    def _ensure_dt_with_tz(obj, tzname_inner):
        local = tz.gettz(tzname_inner)
        if isinstance(obj, str):
            dt = dtparser.parse(obj)
        else:
            dt = obj
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local)
        return dt

    start = _ensure_dt_with_tz(start_str, tzname).astimezone(datetime.timezone.utc)
    end   = _ensure_dt_with_tz(end_str,   tzname).astimezone(datetime.timezone.utc)

    body = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "items": [{"id": "primary"}]
    }
    fb = service.freebusy().query(body=body).execute()
    busy = fb["calendars"]["primary"].get("busy", [])
    return len(busy) == 0, busy

# -------- Natural-language calendar intent parsing --------

_MONTHS_RE = (
    r"(jan(?:uary)?\.?|feb(?:ruary)?\.?|mar(?:ch)?\.?|apr(?:il)?\.?|may\.?|jun(?:e)?\.?|jul(?:y)?\.?|"
    r"aug(?:ust)?\.?|sep(?:t(?:ember)?)?\.?|oct(?:ober)?\.?|nov(?:ember)?\.?|dec(?:ember)?)"
)

_WEEKDAYS = {
    "monday":0,"tuesday":1,"wednesday":2,"thursday":3,
    "friday":4,"saturday":5,"sunday":6
}

_time_range_re = re.compile(
    r"(?P<start>\d{1,2}(:\d{2})?\s*(am|pm)?)\s*(to|-|–|—|until|till|and)\s*"
    r"(?P<end>\d{1,2}(:\d{2})?\s*(am|pm)?)",
    re.IGNORECASE
)

def _wordnum_to_int(s: str):
    s = s.lower().strip().replace("-", " ")
    card_1_19 = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                 "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
    ord_1_19  = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,
                 "eleventh":11,"twelfth":12,"thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,"seventeenth":17,"eighteenth":18,"nineteenth":19}
    tens_card = {"twenty":20,"thirty":30}
    tens_ord = {"twentieth":20,"thirtieth":30}

    if s in card_1_19:
        return card_1_19[s]
    if s in ord_1_19:
        return ord_1_19[s]
    if s in tens_ord:
        return tens_ord[s]

    parts = s.split()
    if len(parts)==2 and parts[0] in tens_card:
        tens = tens_card[parts[0]]
        unit_card = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
        unit_ord  = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9}
        if parts[1] in unit_card:
            return tens + unit_card[parts[1]]
        if parts[1] in unit_ord:
            return tens + unit_ord[parts[1]]
    return None

def _catch_month_day(text: str):
    s = text.lower()
    m = re.search(
        rf"\b{_MONTHS_RE}\b[\s,]+(\d{{1,2}}(?:st|nd|rd|th)?|\w+(?:[-\s]\w+)?)\b",
        s
    )
    if not m:
        return None, None

    month_txt = m.group(1)
    day_txt = m.group(2)

    month_map = {
        "january":1,"jan":1,"jan.":1,
        "february":2,"feb":2,"feb.":2,
        "march":3,"mar":3,"mar.":3,
        "april":4,"apr":4,"apr.":4,
        "may":5,"may.":5,
        "june":6,"jun":6,"jun.":6,
        "july":7,"jul":7,"jul.":7,
        "august":8,"aug":8,"aug.":8,
        "september":9,"sep":9,"sep.":9,"sept":9,"sept.":9,
        "october":10,"oct":10,"oct.":10,
        "november":11,"nov":11,"nov.":11,
        "december":12,"dec":12,"dec.":12
    }

    mnum = month_map.get(month_txt)
    if not mnum:
        return None, None

    d = re.sub(r"(st|nd|rd|th)$", "", day_txt)
    if d.isdigit():
        return mnum, int(d)

    val = _wordnum_to_int(day_txt)
    if val:
        return mnum, val
    return None, None

def _next_weekday(dt_local, target_wd):
    days_ahead = (target_wd - dt_local.weekday()) % 7
    return dt_local + datetime.timedelta(days=days_ahead)

def _parse_day_phrase(text, tzname=LOCAL_TZ_NAME):
    local = tz.gettz(tzname)
    now = datetime.datetime.now(local)
    low = text.lower()

    if "today" in low:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=1)

    if "tomorrow" in low:
        start = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return start, start + datetime.timedelta(days=1)

    for name, wd in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", low):
            day = _next_weekday(now, wd)
            start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, start + datetime.timedelta(days=1)

    mnum, day = _catch_month_day(text)
    if mnum and 1 <= day <= 31:
        year = now.year
        try_date = datetime.datetime(year, mnum, day, tzinfo=local)
        # if it's in the past this year, assume next year
        if try_date < now.replace(hour=0, minute=0, second=0, microsecond=0):
            try_date = datetime.datetime(year+1, mnum, day, tzinfo=local)
        start = try_date.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=1)

    try:
        d = dtparser.parse(text, fuzzy=True, default=now)
        d_local = d.astimezone(local) if d.tzinfo else d.replace(tzinfo=local)
        start = d_local.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + datetime.timedelta(days=1)
    except Exception:
        return None, None

def _parse_time_range(text, base_date_local, tzname=LOCAL_TZ_NAME):
    m = _time_range_re.search(text)
    if not m:
        return None, None

    start_str = m.group("start")
    end_str = m.group("end")

    day_start, _ = _parse_day_phrase(text, tzname=tzname)
    day_base = day_start or base_date_local

    def _combine(date_local, time_str):
        t = dtparser.parse(time_str, fuzzy=True, default=date_local)
        if t.tzinfo:
            return t.astimezone(tz.gettz(tzname))
        return t.replace(tzinfo=tz.gettz(tzname))

    sdt = _combine(day_base, start_str)
    edt = _combine(day_base, end_str)
    if edt <= sdt:
        # assume wrap (like "3pm to 1am")
        edt = edt + datetime.timedelta(hours=12)
    return sdt, edt

def handle_calendar_text(user_text, tzname=LOCAL_TZ_NAME, identity="Aldridge"):
    """
    Interpret natural language calendar queries.
    Returns (handled_bool, tts_summary_string_or_None)
    """
    # Check if text smells like calendar talk
    low = user_text.strip().lower()
    triggers = [
        "what do i have","what's on","whats on","what's happening","whats happening",
        "happening","events","schedule","anything on","busy on","free","available",
        "agenda","calendar"
    ]
    if not any(t in low for t in triggers):
        if not any(w in low for w in ["today","tomorrow"] + list(_WEEKDAYS.keys())):
            try:
                _ = dtparser.parse(user_text, fuzzy=True)
            except Exception:
                return False, None

    local = tz.gettz(tzname)
    base_today = (
        datetime.datetime.now(local)
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )

    # Case 1: "am I free X to Y"
    sdt, edt = _parse_time_range(user_text, base_today, tzname=tzname)
    if sdt and edt:
        try:
            ok, busy = gcal_is_free_between(sdt, edt, tzname=tzname, identity=identity)
            if ok:
                print("You are free in that window.")
                return True, "You are free in that window."
            else:
                print("You are busy in that window:")
                for b in busy:
                    print(f"  - {b['start']} → {b['end']}")
                return True, "You are busy in that window."
        except Exception as e:
            print(f"(Calendar error: {e})")
            return True, "There was a calendar error."

    # Case 2: "what's on Friday / today / Nov 2"
    day_start, day_end = _parse_day_phrase(user_text, tzname=tzname)
    if not day_start:
        try:
            now = datetime.datetime.now(local)
            d = dtparser.parse(user_text, fuzzy=True, default=now)
            d_local = d.astimezone(local) if d.tzinfo else d.replace(tzinfo=local)
            day_start = d_local.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + datetime.timedelta(days=1)
        except Exception:
            return False, None

    try:
        events = gcal_list_upcoming(
            n=100,
            time_min=day_start,
            time_max=day_end,
            tzname=tzname,
            identity=identity
        )

        nice_date = day_start.strftime("%A %b %d, %Y")
        if not events:
            print(f"(No events on {nice_date}.)")
            return True, f"No events on {nice_date}."
        else:
            print(f"Events on {nice_date}:")
            for ev in events:
                print(gcal_format_event(ev, tzname=tzname))
            return True, f"You have {len(events)} event(s) on {nice_date}."
    except Exception as e:
        print(f"(Calendar error: {e})")
        return True, "There was a calendar error."
