import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date
from collections import defaultdict
import urllib.parse as urlparse

# --- External dependency (pip install power-of-10) ---
try:
    from power_of_10.athletes import get_athlete
except Exception:
    get_athlete = None

st.set_page_config(page_title="Power of 10: Season Summary", layout="wide")
st.title("Run MC Performance Summary Web App")
st.caption("Select an athlete (or Whole Group) and a season year.")

# ---------------------- Helpers ----------------------
def darken_color(base_rgb, race_number, min_brightness=0.4):
    factor = max(min_brightness, 1.0 - 0.1 * (race_number - 1))
    return tuple(factor * c for c in base_rgb)

def time_to_seconds(t: str) -> float:
    """Accepts 'mm:ss.xx' or 'ss.xx'. Returns seconds (float)."""
    if not t:
        return float("inf")
    parts = t.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return float("inf")

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%d %b %y")

def seconds_to_mmss(seconds: float) -> str:
    if seconds == float("inf"):
        return "N/A"
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}:{sec:05.2f}"

def format_time(t: float) -> str:
    return seconds_to_mmss(t) if t < float("inf") else "N/A"

# ---------------------- Roster ----------------------
# Default: counts for every year in Whole Group.
# To restrict an athlete to certain years (Whole Group only), add a 4th item: a list/set of years.
#   ("Name", "url", ["events"], {2024, 2025})
default_roster = [ 
    ("Adam Wetton", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=904682", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Alex Friend", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=809748", ["400", "800", "1500", "3000", "5000"], {2025}),
    ("Alex Ibbs", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=359103", ["400", "800", "1500", "3000", "5000"], {2021, 2022, 2023, 2024, 2025}),
    ("Alex Melloy", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=771463", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Alex Pester", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1123918", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Ally Kinlock", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=373303", ["400", "800", "1500", "3000", "5000"], {2021, 2022, 2023, 2024, 2025}),
    ("Andy Power", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=993698", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
#   ("Arthur Starves", "", ["800", "1500"], {2023, 2024, 2025}),
    ("Ben Hamblin", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=898796", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Ben Pattison", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=444384", ["400", "800", "1500", "3000", "5000"], {2022, 2023, 2024, 2025}),
    ("Caleb Stephenson", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=620984", ["400", "800", "1500", "3000", "5000"], {2022, 2023, 2024, 2025}),
    ("Caspar Chang", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1122982", ["400", "800", "1500", "3000", "5000"], {2022, 2023, 2024, 2025}),
    ("Charlie Prestwich", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1042170", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Dom Smith", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1049991", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Dylan Owens", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1000953", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Ethan Isles", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1121514", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Euan Gaskin", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=849041", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Ewan Maxwell", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=561277", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Finley Ball", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=605251", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Finnian Hutchinson", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=390713", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Fintan Kavanagh", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1016954", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Flynn Jennings", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=861171", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("George Thomas", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=773334", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("George Watson", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=112378", ["400", "800", "1500", "3000", "5000"], {2020, 2021, 2022, 2023, 2024, 2025}),
    ("Guy Barnett", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=851839", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Harry Hyde", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=526599", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Harvey Butler", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=714145", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Harvey Hancock", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=645495", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Henry Jonas", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=778952", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Isaac Mould", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1041566", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Izzy Fry", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=352728", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jack Campbell", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=890403", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jack Small", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=760118", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jack White", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=638561", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jake Reynolds", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1203973", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jake Stevens", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=901741", ["400", "800", "1500", "3000", "5000"], {2025}),
    ("James Price", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=880945", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jamie Keir", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=854890", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("John Gordon", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=661771", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jonny Price", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1036113", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Jordan Rowe", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=374986", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("JP Stolberg", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=545008", ["400", "800", "1500", "3000", "5000"], {2021, 2022, 2023, 2024, 2025}),
    ("Kristian Tung", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1103894", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
#   ("Leon Joly", "", ["800", "1500"], {2023, 2024, 2025}),
    ("Lewis Watt", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=642009", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Louis Small", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=760120", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Luca Mastro", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=422814", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Luke Nuttall", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=650578", ["400", "800", "1500", "3000", "5000"], {2022, 2023, 2024, 2025}),
    ("Matthew Stonier", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=516072", ["400", "800", "1500", "3000", "5000"], {2021, 2022, 2023, 2024, 2025}),
    ("Matthew Walton", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=903606", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Max Nicolle", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=562037", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Miles Brown", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=763807", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Nathan Brown", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=354661", ["400", "800", "1500", "3000", "5000"], {2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025}),
    ("Nicklas Shaw", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1173818", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Pat Faulkner", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=602300", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
#   ("Paul Rouyer", "", ["800", "1500"], {2023, 2024, 2025}),
    ("Rhys Simmonds", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1001645", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Robert Smyk", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=831552", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
#   ("Rowan M-I", "", ["800", "1500"], {2023, 2024, 2025}),
    ("Ryan Elston", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=773467", ["400", "800", "1500", "3000", "5000"], {2022, 2023, 2024, 2025}),
    ("Sam Goodchild", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=567114", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Sam Hodgson", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=718408", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Sam Wiggins", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=750081", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Ted Ash", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=1056195", ["400", "800", "1500", "3000", "5000"], {2024, 2025}),
    ("Tom Chadwick", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=718514", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Tom Kimber", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=602065", ["400", "800", "1500", "3000", "5000"], {2023, 2024, 2025}),
    ("Tom Mortimer", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=527528", ["400", "800", "1500", "3000", "5000"], {2020, 2021, 2022, 2023, 2024, 2025}),
    ("Tom Rutherford", "https://www.thepowerof10.info/athletes/profile.aspx?athleteid=684571", ["400", "800", "1500", "3000", "5000"], {2021, 2022, 2023, 2024, 2025}),
]

# Normalize roster to a uniform dict shape and never unpack tuples elsewhere
norm_roster = []
for item in default_roster:
    if len(item) < 3:
        raise ValueError("Roster rows must have at least 3 items: (name, url, events[, years])")
    name = item[0]
    url = item[1]
    events = item[2]
    years = None
    if len(item) >= 4 and item[3] is not None:
        # Accept set/list/tuple of ints/strings and coerce to set of ints
        years = {int(y) for y in item[3]}
    norm_roster.append({"name": name, "url": url, "events": events, "years": years})

# ---------------------- UI: selection ----------------------
athlete_names = ["Whole Group"] + [x["name"] for x in norm_roster]
selected_athlete_name = st.selectbox("Select Athlete", athlete_names)

selected_url, selected_events = None, []
if selected_athlete_name != "Whole Group":
    rec = next(r for r in norm_roster if r["name"] == selected_athlete_name)
    selected_url = rec["url"]
    selected_events = [e for e in rec["events"] if ("XC" not in e.upper()) and ("PARKRUN" not in e.upper())]
    selected_event = st.selectbox("Select Event", selected_events)
else:
    selected_event = None

year = st.number_input("Season year", min_value=2010, max_value=2100, value=2025, step=1)
run_btn = st.button("Run", type="primary")

# ---------------------- Data Fetch ----------------------
@st.cache_data(show_spinner=False)
def fetch_athlete(athlete_id: str):
    if get_athlete is None:
        st.error("The 'power-of-10' package is not installed. Run: pip install power-of-10")
        st.stop()
    return get_athlete(athlete_id)

# ---------------------- Plot builder ----------------------
def make_interactive_figure(athlete_name, age_group, event_label, dates, times, orders):
    base_rgb = (0.56, 0.93, 0.56)  # lightgreen
    colors = [
        "rgb({},{},{})".format(int(255*c[0]), int(255*c[1]), int(255*c[2]))
        for c in [darken_color(base_rgb, o) for o in orders]
    ]
    customdata = [{"tstr": seconds_to_mmss(float(t))} for t in times]

    fig = go.Figure()

    if dates:
        start_year = min(d.year for d in dates)
        end_year = max(d.year for d in dates)
        shapes = []
        for Y in range(start_year, end_year + 1):
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=date(Y-1,10,1), x1=date(Y,3,31),
                               y0=0, y1=1, fillcolor="lightblue",
                               opacity=0.25, line_width=0))
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=date(Y,4,1), x1=date(Y,9,30),
                               y0=0, y1=1, fillcolor="khaki",
                               opacity=0.25, line_width=0))
        fig.update_layout(shapes=shapes)

    fig.add_trace(
        go.Scattergl(
            x=dates,
            y=times,
            mode="markers",
            marker=dict(
                size=[10 if o not in [6,7,8] else 16 for o in orders],
                line=dict(width=1, color="black"),
                color=colors
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                + event_label + ": %{customdata.tstr}<extra></extra>"
            ),
            name=f"{event_label}",
        )
    )

    fig.update_yaxes(title_text="Time (s)", autorange="reversed")
    fig.update_xaxes(title_text="Date")
    title = f"{athlete_name} ({age_group}) — {event_label}m"
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=60, b=40), height=520)
    return fig

# ---------------------- Whole Group mode ----------------------
if run_btn and selected_athlete_name == "Whole Group":
    # Per-year aggregations
    years_seen = set()
    total_pbs_year = defaultdict(int)
    ranks_year = defaultdict(lambda: {"top10": 0, "top25": 0, "top50": 0, "top100": 0})
    pb_counts_per_athlete_year = defaultdict(int)  # (athlete, year) -> PB count (running-best)
    improver_athletes_year = defaultdict(set)      # year -> set(athletes with >5% improvement)
    fastest_year = defaultdict(lambda: {
        ("Male","400"): float("inf"), ("Female","400"): float("inf"),
        ("Male","800"): float("inf"), ("Female","800"): float("inf"),
        ("Male","1500"): float("inf"), ("Female","1500"): float("inf"),
    })

    # Selected year (single-season) summary
    total_perfs_selected = 0
    total_pbs_selected = 0
    fastest_selected = {
        "Male": {"400": (float("inf"), ""), "800": (float("inf"), ""), "1500": (float("inf"), "")},
        "Female": {"400": (float("inf"), ""), "800": (float("inf"), ""), "1500": (float("inf"), "")},
    }
    ranking_rows_selected = []
    athlete_pb_counts_selected = defaultdict(int)
    pb_improved_over_5pct_selected = []

    progress = st.progress(0.0)
    for i, rec in enumerate(norm_roster, start=1):
        name, url, years_allowed = rec["name"], rec["url"], rec["years"]
        try:
            athlete_id = urlparse.parse_qs(urlparse.urlparse(url).query).get("athleteid", [None])[0]
            profile = fetch_athlete(athlete_id)
        except Exception as e:
            st.error(f"Failed to fetch {name}: {e}")
            progress.progress(i/len(norm_roster))
            continue

        # Filter (exclude XC/parkrun)
        performances = [
            p for p in profile.get("performances", [])
            if "XC" not in (p.get("event") or "").upper()
            and "PARKRUN" not in (p.get("event") or "").upper()
        ]
        ranks = profile.get("rankings", []) if isinstance(profile.get("rankings", []), list) else []
        ranks = [r for r in ranks if "XC" not in (r.get("event") or "").upper() and "PARKRUN" not in (r.get("event") or "").upper()]

        gender_key = profile.get("gender", "Male")
        if gender_key not in ("Male", "Female"):
            gender_key = "Male"

        # Group performances by event
        per_event = defaultdict(list)
        for perf in performances:
            try:
                ev = perf.get("event")
                d = parse_date(perf.get("date"))
                t = time_to_seconds(perf.get("value"))
                if t < float("inf"):
                    per_event[ev].append((d, t))
                    years_seen.add(d.year)
            except Exception:
                continue
        for ev in list(per_event.keys()):
            per_event[ev].sort(key=lambda x: x[0])

        # Per-year PBs, improvers, fastest (respect roster year limits)
        for ev, lst in per_event.items():
            # Running PBs and PB counts per athlete-year
            pb_by_year = defaultdict(int)
            best_so_far = float('inf')
            by_year_values = defaultdict(list)

            for d, t in lst:
                Y = d.year
                if years_allowed is not None and Y not in years_allowed:
                    continue
                by_year_values[Y].append(t)
                if t < best_so_far:
                    best_so_far = t
                    pb_by_year[Y] += 1

                # Fastest per year for canonical
                if ev in ("400", "800", "1500"):
                    cur = fastest_year[Y][(gender_key, ev)]
                    if t < cur:
                        fastest_year[Y][(gender_key, ev)] = t

            for Y, cnt in pb_by_year.items():
                total_pbs_year[Y] += cnt
                pb_counts_per_athlete_year[(name, Y)] += cnt

            # >5% improvement: compare each year's best vs rolling best-before
            best_before = float('inf')
            for Y in sorted(by_year_values.keys()):
                in_year_best = min(by_year_values[Y])
                if best_before < float('inf') and in_year_best < best_before:
                    pct = (best_before - in_year_best) / best_before * 100
                    if pct > 5:
                        improver_athletes_year[Y].add(name)
                best_before = min(best_before, in_year_best)

            # Selected year tables
            if years_allowed is None or int(year) in years_allowed:
                yr = int(year)
                year_times = [t for d, t in lst if d.year == yr]
                # PBs in selected year via running best
                pb_by_year_sel = defaultdict(int)
                best_so_far_sel = float('inf')
                for d, t in lst:
                    if t < best_so_far_sel:
                        best_so_far_sel = t
                        pb_by_year_sel[d.year] += 1
                total_pbs_selected += pb_by_year_sel.get(yr, 0)
                athlete_pb_counts_selected[name] += pb_by_year_sel.get(yr, 0)
                total_perfs_selected += sum(1 for d, _ in lst if d.year == yr)

                # Fastest (selected year)
                if ev in ("400", "800", "1500") and year_times:
                    candidate = min(year_times)
                    cur_best, _ = fastest_selected[gender_key][ev]
                    if candidate < cur_best:
                        fastest_selected[gender_key][ev] = (candidate, name)

                # >5% improvers (selected year)
                prev_best = min([t for d, t in lst if d.year < yr], default=float("inf"))
                if year_times and prev_best < float("inf"):
                    best_in_year = min(year_times)
                    if best_in_year < prev_best:
                        pct = (prev_best - best_in_year) / prev_best * 100
                        if pct > 5:
                            pb_improved_over_5pct_selected.append({
                                "Athlete": name, "Event": ev, "% Improvement": f"{pct:.1f}%"
                            })

        # Rankings buckets per year
        for r in ranks:
            try:
                Y = int(r.get("year"))
                if years_allowed is not None and Y not in years_allowed:
                    continue
                rank_val = int(r.get("rank"))
                if 1 <= rank_val <= 100:
                    ranks_year[Y]["top100"] += 1
                    if rank_val <= 50: ranks_year[Y]["top50"] += 1
                    if rank_val <= 25: ranks_year[Y]["top25"] += 1
                    if rank_val <= 10: ranks_year[Y]["top10"] += 1
                if (years_allowed is None or int(year) in years_allowed) and Y == int(year):
                    ranking_rows_selected.append({"Athlete": name, "Event": r.get("event"), "Rank": rank_val})
            except Exception:
                pass

        progress.progress(i/len(norm_roster))

    # Trend charts
    years_sorted = sorted(years_seen)
    if years_sorted:
        st.subheader("Group Trends (all available years)")

        # PBs per year
        fig_pbs = go.Figure()
        fig_pbs.add_trace(go.Scatter(x=years_sorted, y=[total_pbs_year.get(Y, 0) for Y in years_sorted],
                                     mode='lines+markers', name='PBs'))
        fig_pbs.update_layout(title="Number of PBs per Year", xaxis_title="Year", yaxis_title="PBs")
        st.plotly_chart(fig_pbs, use_container_width=True)

        # Rankings counts per year
        y100 = [ranks_year[Y]["top100"] for Y in years_sorted]
        y50  = [ranks_year[Y]["top50"]  for Y in years_sorted]
        y25  = [ranks_year[Y]["top25"]  for Y in years_sorted]
        y10  = [ranks_year[Y]["top10"]  for Y in years_sorted]
        fig_ranks = go.Figure()
        fig_ranks.add_trace(go.Scatter(x=years_sorted, y=y100, mode='lines+markers', name='Top 100'))
        fig_ranks.add_trace(go.Scatter(x=years_sorted, y=y50,  mode='lines+markers', name='Top 50'))
        fig_ranks.add_trace(go.Scatter(x=years_sorted, y=y25,  mode='lines+markers', name='Top 25'))
        fig_ranks.add_trace(go.Scatter(x=years_sorted, y=y10,  mode='lines+markers', name='Top 10'))
        fig_ranks.update_layout(title="UK Rankings Counts by Year", xaxis_title="Year", yaxis_title="Count")
        st.plotly_chart(fig_ranks, use_container_width=True)

        # # athletes with >5 PBs per year
        over5_counts = []
        for Y in years_sorted:
            athletes_over5 = {ath for (ath, y) in pb_counts_per_athlete_year if y == Y and pb_counts_per_athlete_year[(ath, y)] > 5}
            over5_counts.append(len(athletes_over5))
        fig_over5 = go.Figure()
        fig_over5.add_trace(go.Scatter(x=years_sorted, y=over5_counts, mode='lines+markers', name='Athletes >5 PBs'))
        fig_over5.update_layout(title="Athletes with >5 PBs per Year", xaxis_title="Year", yaxis_title="# Athletes")
        st.plotly_chart(fig_over5, use_container_width=True)

        # # athletes with >5% improvement per year
        impr_counts = [len(improver_athletes_year.get(Y, set())) for Y in years_sorted]
        fig_impr = go.Figure()
        fig_impr.add_trace(go.Scatter(x=years_sorted, y=impr_counts, mode='lines+markers', name='Athletes >5% Improvement'))
        fig_impr.update_layout(title="Athletes with >5% Improvement per Year", xaxis_title="Year", yaxis_title="# Athletes")
        st.plotly_chart(fig_impr, use_container_width=True)

        # Fastest time lines per event & gender

        for ev in ("400","800","1500"):
            fig_fast = go.Figure()
            male_series = [fastest_year[Y][("Male", ev)] for Y in years_sorted]
            female_series = [fastest_year[Y][("Female", ev)] for Y in years_sorted]
            male_series = [v if v != float('inf') else None for v in male_series]
            female_series = [v if v != float('inf') else None for v in female_series]

            male_custom  = [seconds_to_mmss(v) if v is not None else None for v in male_series]
            female_custom= [seconds_to_mmss(v) if v is not None else None for v in female_series]

            fig_fast.add_trace(go.Scatter(
                x=years_sorted, y=male_series, mode='lines+markers', name='Male',
                customdata=male_custom,
                hovertemplate="<b>%{x}</b><br>Male: %{customdata}<extra></extra>"
            ))
            fig_fast.add_trace(go.Scatter(
                x=years_sorted, y=female_series, mode='lines+markers', name='Female',
                customdata=female_custom,
                hovertemplate="<b>%{x}</b><br>Female: %{customdata}<extra></extra>"
            ))

            all_secs = [v for v in (male_series + female_series) if v is not None]
            if all_secs:
                y_min, y_max = min(all_secs), max(all_secs)
                span = y_max - y_min
                if span <= 10:        step = 1
                elif span <= 20:      step = 2
                elif span <= 60:      step = 5
                elif span <= 120:     step = 10
                else:                 step = 15
                start = int((y_min // step) * step)
                ticks = list(range(start, int(y_max) + step, step))
                fig_fast.update_yaxes(
                    tickmode="array",
                    tickvals=ticks,
                    ticktext=[seconds_to_mmss(v) for v in ticks],
                    title_text="Time (mm:ss)"
                )
            else:
                fig_fast.update_yaxes(title_text="Time (mm:ss)")

            fig_fast.update_layout(
                title=f"Fastest {ev}m by Year (lower is better)",
                xaxis_title="Year"
            )
            st.plotly_chart(fig_fast, use_container_width=True)
    
  #      for ev in ("400","800","1500"):
  #          fig_fast = go.Figure()
  #          male_series = [fastest_year[Y][("Male", ev)] for Y in years_sorted]
  #          female_series = [fastest_year[Y][("Female", ev)] for Y in years_sorted]
  #          male_series = [v if v != float('inf') else None for v in male_series]
  #          female_series = [v if v != float('inf') else None for v in female_series]
  #          fig_fast.add_trace(go.Scatter(x=years_sorted, y=female_series, mode='lines+markers', name='Female'))
  #          fig_fast.add_trace(go.Scatter(x=years_sorted, y=male_series, mode='lines+markers', name='Male'))
  #          fig_fast.update_layout(title=f"Fastest {ev}m by Year (lower is better)", xaxis_title="Year", yaxis_title="Time (s)")
  #          st.plotly_chart(fig_fast, use_container_width=True)
    else:
        st.warning("No data found to plot across years.")

    # Selected-year summary tables
    st.subheader(f"Selected Year Summary — {int(year)}")
    c1, c2 = st.columns(2)
    c1.metric("Total Performances", total_perfs_selected)
    c2.metric("Total PBs", total_pbs_selected)

    fast_rows_sel = []
    for g in ("Male", "Female"):
        for ev in ("400", "800", "1500"):
            t, n = fastest_selected[g][ev]
            fast_rows_sel.append({"Gender": g, "Event": ev, "Athlete": n, "Time": format_time(t)})
    st.markdown("**Fastest Times (by gender & event)**")
    st.dataframe(fast_rows_sel, use_container_width=True)

    def bucketize(rows):
        b10, b25, b50, b100 = [], [], [], []
        for r in rows:
            rank = r["Rank"]
            entry = {"Athlete": r["Athlete"], "Event": r["Event"], "Rank": rank}
            if rank <= 10: b10.append(entry)
            elif rank <= 25: b25.append(entry)
            elif rank <= 50: b50.append(entry)
            elif rank <= 100: b100.append(entry)
        for b in (b10, b25, b50, b100):
            b.sort(key=lambda x: x["Rank"])
        return b10, b25, b50, b100

    if ranking_rows_selected:
        top10, top25, top50, top100 = bucketize(ranking_rows_selected)
        st.markdown("**UK Rankings — Top 10**")
        st.dataframe(top10 or [{"Athlete":"—","Event":"—","Rank":"—"}], use_container_width=True)
        st.markdown("**UK Rankings — 11–25**")
        st.dataframe(top25 or [{"Athlete":"—","Event":"—","Rank":"—"}], use_container_width=True)
        st.markdown("**UK Rankings — 26–50**")
        st.dataframe(top50 or [{"Athlete":"—","Event":"—","Rank":"—"}], use_container_width=True)
        st.markdown("**UK Rankings — 51–100**")
        st.dataframe(top100 or [{"Athlete":"—","Event":"—","Rank":"—"}], use_container_width=True)
    else:
        st.info("No ranking data found for the selected year.")

    pb_over_5_sel = [{"Athlete": n, "Total PBs": c} for n, c in athlete_pb_counts_selected.items() if c > 5]
    pb_over_5_sel.sort(key=lambda x: x["Total PBs"], reverse=True)
    st.markdown("**Athletes with >5 PBs in the year**")
    if pb_over_5_sel:
        st.dataframe(pb_over_5_sel, use_container_width=True)
    else:
        st.info("No athletes with more than 5 PBs recorded in the selected year.")

    st.markdown("**Athletes with PB Improvements >5% (year vs previous best)**")
    if pb_improved_over_5pct_selected:
        def pct_val(row):
            try: return float(row["% Improvement"].rstrip("%"))
            except Exception: return 0.0
        pb_improved_over_5pct_selected.sort(key=pct_val, reverse=True)
        st.dataframe(pb_improved_over_5pct_selected, use_container_width=True)
    else:
        st.info("No athletes improved by more than 5% this year.")

# ---------------------- Individual mode ----------------------
elif run_btn and selected_url:
    try:
        athlete_id = urlparse.parse_qs(urlparse.urlparse(selected_url).query).get("athleteid", [None])[0]
        profile = fetch_athlete(athlete_id)
    except Exception as e:
        st.error(f"Failed to fetch {selected_athlete_name}: {e}")
        st.stop()

    performances = [
        p for p in profile.get("performances", [])
        if "XC" not in (p.get("event") or "").upper()
        and "PARKRUN" not in (p.get("event") or "").upper()
    ]
    age_group = profile.get("age_group", "")

    # Selected event performances
    event_perfs = []
    for perf in performances:
        if perf.get("event") == selected_event:
            try:
                d = parse_date(perf.get("date"))
                t = time_to_seconds(perf.get("value"))
                if t < float("inf"):
                    event_perfs.append((d, t))
            except Exception:
                continue
    event_perfs.sort(key=lambda x: x[0])

    # Totals for this athlete & year
    yr = int(year)
    total_perfs = sum(1 for d, _ in event_perfs if d.year == yr)
    pb_by_year = defaultdict(int)
    best_so_far = float("inf")
    for d, t in event_perfs:
        if t < best_so_far:
            best_so_far = t
            pb_by_year[d.year] += 1
    total_pbs = pb_by_year.get(yr, 0)

    year_times = [t for d, t in event_perfs if d.year == yr]
    best_time_val = min(year_times) if year_times else float("inf")
    mean_time_val = (sum(year_times) / len(year_times)) if year_times else float("inf")

    # Metrics above the graph
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Performances (year)", total_perfs)
    c2.metric("Total PBs (year)", total_pbs)
    c3.metric("Best (year)", seconds_to_mmss(best_time_val) if year_times else "N/A")
    c4.metric("Mean (year)", seconds_to_mmss(mean_time_val) if year_times else "N/A")

    # Plot + table
    if event_perfs:
        dates, times, orders = [], [], []
        yearly_counts = defaultdict(int)
        for d, t in event_perfs:
            yearly_counts[d.year] += 1
            dates.append(d)
            times.append(t)
            orders.append(yearly_counts[d.year])

        fig = make_interactive_figure(selected_athlete_name, age_group, selected_event, dates, times, orders)
        if times:  # only if we actually have data
            unique_secs = sorted(set(int(t) for t in times))
            fig.update_yaxes(
                tickmode="array",
                tickvals=unique_secs,
                ticktext=[seconds_to_mmss(v) for v in unique_secs],
                title_text="Time (mm:ss)"
            )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        table_rows = []
        for (d, t), o in zip(event_perfs, orders):
            table_rows.append({
                "Date": d.strftime("%d %b %Y"),
                "Event": selected_event,
                "Time": seconds_to_mmss(t),
                "Year": d.year,
                "Race # in Year": o,
            })
        st.markdown("**Performances**")
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No performances found for this event.")


