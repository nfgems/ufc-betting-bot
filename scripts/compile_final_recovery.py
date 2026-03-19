"""Compile final recovery and unresolved CSV files from all sources."""
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_RECOVERY = REPO_ROOT / "data/raw/method_odds/historical_method_odds_manual_recovery_20260319.csv"
OUTPUT_UNRESOLVED = REPO_ROOT / "data/raw/method_odds/historical_method_odds_manual_unresolved_20260319.csv"

HEADERS_ROW = [
    "event_date", "fighter_a", "fighter_b", "event_title",
    "source", "source_url", "captured_at",
    "a_ko_odds_prob", "a_sub_odds_prob", "a_dec_odds_prob",
    "b_ko_odds_prob", "b_sub_odds_prob", "b_dec_odds_prob",
]


def am(odds):
    """American odds to implied probability."""
    if odds > 0:
        return round(100.0 / (odds + 100.0), 6)
    return round(abs(odds) / (abs(odds) + 100.0), 6)


# Load Wayback-sourced recoveries from checkpoint
ckpt_path = REPO_ROOT / "data/raw/method_odds/manual_recovery_checkpoint.json"
with open(ckpt_path) as f:
    ckpt = json.load(f)
wayback_recovered = ckpt["recovered"]

# New recoveries from FanDuel/web sources
web_recovered = [
    # UFC 281: Zhang Weili vs Carla Esparza (FanDuel via boardroom.tv)
    {"event_date": "2022-11-12", "fighter_a": "Zhang Weili", "fighter_b": "Carla Esparza",
     "event_title": "UFC 281: Adesanya vs. Pereira",
     "source": "fanduel_sportsbook",
     "source_url": "https://boardroom.tv/carla-esparza-zhang-weili-odds-props-prediction-ufc-281/",
     "captured_at": "2022-11-12T00:00:00Z",
     "a_ko_odds_prob": am(125), "a_sub_odds_prob": am(1100), "a_dec_odds_prob": am(195),
     "b_ko_odds_prob": am(1500), "b_sub_odds_prob": am(1500), "b_dec_odds_prob": am(500)},
    # UFC 281: Poirier vs Chandler (FanDuel via boardroom.tv)
    {"event_date": "2022-11-12", "fighter_a": "Dustin Poirier", "fighter_b": "Michael Chandler",
     "event_title": "UFC 281: Adesanya vs. Pereira",
     "source": "fanduel_sportsbook",
     "source_url": "https://boardroom.tv/dustin-poirier-michael-chandler-odds-props-prediction-ufc-281/",
     "captured_at": "2022-11-12T00:00:00Z",
     "a_ko_odds_prob": am(125), "a_sub_odds_prob": am(700), "a_dec_odds_prob": am(380),
     "b_ko_odds_prob": am(380), "b_sub_odds_prob": am(1000), "b_dec_odds_prob": am(950)},
    # UFC 282: Topuria vs Mitchell (BetOnline via flurrysports.org)
    {"event_date": "2022-12-10", "fighter_a": "Ilia Topuria", "fighter_b": "Bryce Mitchell",
     "event_title": "UFC 282: Blachowicz vs. Ankalaev",
     "source": "betonline_sportsbook",
     "source_url": "https://flurrysports.org/bryce-mitchell-vs-ilia-topuria-prediction-ufc-282-betting-odds-picks/",
     "captured_at": "2022-12-10T00:00:00Z",
     "a_ko_odds_prob": am(250), "a_sub_odds_prob": am(600), "a_dec_odds_prob": am(260),
     "b_ko_odds_prob": am(1200), "b_sub_odds_prob": am(500), "b_dec_odds_prob": am(250)},
    # UFC 287: Adesanya vs Pereira 2 (FanDuel via boardroom.tv)
    {"event_date": "2023-04-08", "fighter_a": "Israel Adesanya", "fighter_b": "Alex Pereira",
     "event_title": "UFC 287: Pereira vs. Adesanya 2",
     "source": "fanduel_sportsbook",
     "source_url": "https://boardroom.tv/adesanya-vs-pereira-prediction-odds-ufc-287/",
     "captured_at": "2023-04-08T00:00:00Z",
     "a_ko_odds_prob": am(440), "a_sub_odds_prob": am(2200), "a_dec_odds_prob": am(160),
     "b_ko_odds_prob": am(230), "b_sub_odds_prob": am(2200), "b_dec_odds_prob": am(500)},
    # UFC 290: Volkanovski vs Rodriguez (FanDuel via boardroom.tv)
    {"event_date": "2023-07-08", "fighter_a": "Alexander Volkanovski", "fighter_b": "Yair Rodriguez",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://boardroom.tv/volkanovski-vs-rodriguez-prediction-odds-ufc-290/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(260), "a_sub_odds_prob": am(1400), "a_dec_odds_prob": am(105),
     "b_ko_odds_prob": am(550), "b_sub_odds_prob": am(1000), "b_dec_odds_prob": am(1000)},
    # UFC 290: Pantoja vs Moreno (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Alexandre Pantoja", "fighter_b": "Brandon Moreno",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/brandon-moreno-vs-alexandre-pantoja-prediction-odds-best-bet-ufc-290-01h4kmzksgsa/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(1100), "a_sub_odds_prob": am(460), "a_dec_odds_prob": am(460),
     "b_ko_odds_prob": am(300), "b_sub_odds_prob": am(1000), "b_dec_odds_prob": am(160)},
    # UFC 290: Du Plessis vs Whittaker (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Dricus Du Plessis", "fighter_b": "Robert Whittaker",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/robert-whittaker-vs-dricus-du-plessis-prediction-odds-best-bet-ufc-290-01h4nr886ws3/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(500), "a_sub_odds_prob": am(1500), "a_dec_odds_prob": am(1000),
     "b_ko_odds_prob": am(135), "b_sub_odds_prob": am(1300), "b_dec_odds_prob": am(190)},
    # UFC 290: Hooker vs Turner (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Dan Hooker", "fighter_b": "Jalin Turner",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/jalin-turner-vs-dan-hooker-prediction-odds-best-bet-ufc-290-01h4p84kwh1h",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(600), "a_sub_odds_prob": am(1600), "a_dec_odds_prob": am(500),
     "b_ko_odds_prob": am(195), "b_sub_odds_prob": am(500), "b_dec_odds_prob": am(250)},
    # UFC 290: Nickal vs Woodburn (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Bo Nickal", "fighter_b": "Val Woodburn",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/bo-nickal-val-woodburn-prediction-odds-best-bet-ufc-290-01h4rgng4j0f/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(290), "a_sub_odds_prob": am(-280), "a_dec_odds_prob": am(1700),
     "b_ko_odds_prob": am(2000), "b_sub_odds_prob": am(6000), "b_dec_odds_prob": am(2600)},
    # UFC 290: Lawler vs Price (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Robbie Lawler", "fighter_b": "Niko Price",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/robbie-lawler-vs-niko-price-prediction-odds-best-bet-ufc-290-01h4p9sqt0wr/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(500), "a_sub_odds_prob": am(2400), "a_dec_odds_prob": am(480),
     "b_ko_odds_prob": am(125), "b_sub_odds_prob": am(950), "b_dec_odds_prob": am(310)},
    # UFC 290: Taira vs Chairez (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Tatsuro Taira", "fighter_b": "Edgar Chairez",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/tatsuro-taira-edgar-chairez-prediction-odds-best-bet-ufc-290-01h4rf333gcc/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(750), "a_sub_odds_prob": am(-190), "a_dec_odds_prob": am(430),
     "b_ko_odds_prob": am(1500), "b_sub_odds_prob": am(2500), "b_dec_odds_prob": am(1100)},
    # UFC 290: Gomes vs Jauregui (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Denise Gomes", "fighter_b": "Yazmin Jauregui",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/yazmin-jauregui-denise-gomes-prediction-odds-best-bet-ufc-290-01h4rbxze9fm/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(1000), "a_sub_odds_prob": am(1900), "a_dec_odds_prob": am(600),
     "b_ko_odds_prob": am(195), "b_sub_odds_prob": am(1500), "b_dec_odds_prob": am(110)},
    # UFC 290: Menifield vs Crute (FanDuel)
    {"event_date": "2023-07-08", "fighter_a": "Alonzo Menifield", "fighter_b": "Jimmy Crute",
     "event_title": "UFC 290: Volkanovski vs. Rodriguez",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.fanduel.com/research/theduel/jimmy-crute-alonzo-menifield-prediction-odds-best-bet-ufc-290-01h4rdp082kt/",
     "captured_at": "2023-07-08T00:00:00Z",
     "a_ko_odds_prob": am(190), "a_sub_odds_prob": am(1800), "a_dec_odds_prob": am(600),
     "b_ko_odds_prob": am(310), "b_sub_odds_prob": am(350), "b_dec_odds_prob": am(500)},
    # Noche UFC: Grasso vs Shevchenko (FanDuel via legalsportsreport.com)
    {"event_date": "2023-09-16", "fighter_a": "Alexa Grasso", "fighter_b": "Valentina Shevchenko",
     "event_title": "UFC Fight Night: Grasso vs. Shevchenko 2",
     "source": "fanduel_sportsbook",
     "source_url": "https://www.legalsportsreport.com/136402/noche-ufc-odds-alexa-grasso-vs-valentina-shevchenko-2-2023/",
     "captured_at": "2023-09-13T00:00:00Z",
     "a_ko_odds_prob": am(700), "a_sub_odds_prob": am(350), "a_dec_odds_prob": am(200),
     "b_ko_odds_prob": am(800), "b_sub_odds_prob": am(1000), "b_dec_odds_prob": am(175)},
]

# Combine
all_recovered = wayback_recovered + web_recovered

# Write recovery CSV
with open(OUTPUT_RECOVERY, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS_ROW)
    writer.writeheader()
    for rec in all_recovered:
        row = {}
        for col in HEADERS_ROW:
            val = rec.get(col, "")
            if isinstance(val, float):
                row[col] = round(val, 6)
            else:
                row[col] = val
        writer.writerow(row)

print(f"Recovery file written: {len(all_recovered)} rows")
for r in all_recovered:
    src = r.get("source", "")
    print(f"  {r['event_date']}: {r['fighter_a']} vs {r['fighter_b']} ({src})")

# Unresolved fights
unresolved = [
    {"fight": "2014-06-28: Joe Ellenberger vs James Moontasri",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800",
     "why_unresolved": "No method prop odds posted on BFO for this undercard fight (confirmed by Playwright scraper and Wayback)"},
    {"fight": "2014-06-28: Cody Gibson vs Johnny Bedford",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800",
     "why_unresolved": "No method prop odds posted on BFO for this undercard fight"},
    {"fight": "2014-06-28: Marcelo Guimaraes vs Andy Enz",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800",
     "why_unresolved": "No method prop odds posted on BFO for this undercard fight"},
    {"fight": "2014-06-28: Ray Borg vs Shane Howell",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800",
     "why_unresolved": "No method prop odds posted on BFO for this undercard fight"},
    {"fight": "2014-06-28: Aleksei Oleinik vs Anthony Hamilton",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-44-swanson-vs-stephens-800",
     "why_unresolved": "No method prop odds posted on BFO for this undercard fight"},
    {"fight": "2015-07-18: Joanne Wood vs Cortney Casey",
     "attempted_sources": "BFO existing data",
     "urls_checked": "",
     "why_unresolved": "ALIAS_RESOLVED: jansen88 lists Joanne Wood but actual fighter is Joanne Calderwood. Data exists under Calderwood."},
    {"fight": "2016-01-02: Justine Kish vs Nina Nunes",
     "attempted_sources": "BFO existing data",
     "urls_checked": "",
     "why_unresolved": "ALIAS_RESOLVED: Nina Nunes listed as Nina Ansaroff on BFO (married name change). Data exists under Ansaroff."},
    {"fight": "2015-05-30: Rony Jason vs Damon Jackson",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-67-alves-vs-condit-957",
     "why_unresolved": "No Wayback snapshots with method props. BFO live redirects to homepage."},
    {"fight": "2015-09-26: Mizuto Hirota vs Teruto Ishihara",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-75-nelson-vs-barnett-1003",
     "why_unresolved": "No Wayback snapshots with method props. BFO live redirects to homepage."},
    {"fight": "2016-09-17: Alejandro Perez vs Albert Morales",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-94-poirier-vs-johnson-1160",
     "why_unresolved": "No Wayback snapshots with method props. BFO live redirects to homepage."},
    {"fight": "2016-11-19: Cezar Ferreira vs Jack Hermansson",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185",
     "why_unresolved": "No Wayback snapshots with method props. BFO live redirects to homepage."},
    {"fight": "2016-11-19: Johnny Eduardo vs Manvel Gamburyan",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185",
     "why_unresolved": "No Wayback snapshots with method props. BFO live redirects to homepage."},
    {"fight": "2016-11-19: Francimar Barroso vs Darren Stewart",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-100-bader-vs-nogueira-1185",
     "why_unresolved": "No Wayback snapshots with method props. BFO live redirects to homepage."},
    {"fight": "2017-06-17: Li Jingliang vs Frank Camacho",
     "attempted_sources": "BFO Playwright, Wayback Machine, Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-111-holm-vs-correia-1277",
     "why_unresolved": "Wayback had 4 fighters but no match for Jingliang/Camacho. Undercard fight."},
    {"fight": "2022-02-05: Jailton Almeida vs Danilo Marques",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-hermansson-vs-strickland-2285",
     "why_unresolved": "BFO redirects all historical events. No Wayback snapshots. Undercard debut fight."},
    {"fight": "2023-01-14: Mateusz Rebecki vs Nick Fiore",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-strickland-vs-imavov-2457",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-02-04: Dooho Choi vs Kyle Nelson",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-lewis-vs-spivac-2472",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-02-04: JeongYeong Lee vs Yizha",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-lewis-vs-spivac-2472",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-03-11: Ariane Lipski vs JJ Aldrich",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-yan-vs-dvalishvili-2490",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-06-17: Dan Argueta vs Ronnie Lawrence",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-vettori-vs-cannonier-2545",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-06-24: Austen Lane vs Justin Tafa",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-emmett-vs-topuria-2546",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-08-05: Billy Quarantillo vs Damon Jackson",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561",
     "why_unresolved": "BFO redirects. No Wayback snapshots. No web article method props found."},
    {"fight": "2023-08-05: Sean Woodson vs Dennis Buzukja",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-08-05: Assu Almabayev vs Ode Osbourne",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/ufc-fight-night-sandhagen-vs-font-2561",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
    {"fight": "2023-09-16: Edgar Chairez vs Daniel Lacerda",
     "attempted_sources": "BFO (redirects), Wayback (no snapshots), Web Search",
     "urls_checked": "https://www.bestfightodds.com/events/noche-ufc-grasso-vs-shevchenko-2-2580",
     "why_unresolved": "BFO redirects. No Wayback snapshots. Undercard fight."},
]

# Write recovery CSV
with open(OUTPUT_RECOVERY, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS_ROW)
    writer.writeheader()
    for rec in all_recovered:
        row = {}
        for col in HEADERS_ROW:
            val = rec.get(col, "")
            if isinstance(val, float):
                row[col] = round(val, 6)
            else:
                row[col] = val
        writer.writerow(row)

print(f"Recovery file: {OUTPUT_RECOVERY.name} ({len(all_recovered)} rows)")
for r in all_recovered:
    print(f"  {r['event_date']}: {r['fighter_a']} vs {r['fighter_b']} ({r.get('source','')})")

# Write unresolved CSV
with open(OUTPUT_UNRESOLVED, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["fight", "attempted_sources", "urls_checked", "why_unresolved"])
    writer.writeheader()
    for rec in unresolved:
        writer.writerow(rec)

alias_ct = sum(1 for u in unresolved if "ALIAS_RESOLVED" in u["why_unresolved"])
print(f"\nUnresolved file: {OUTPUT_UNRESOLVED.name} ({len(unresolved)} rows)")
print(f"  Alias-resolved: {alias_ct}")
print(f"  Truly unresolved: {len(unresolved) - alias_ct}")
print(f"\n=== FINAL SUMMARY ===")
print(f"Total recovered: {len(all_recovered)} / 45")
print(f"  From Wayback Machine (BFO): {len(wayback_recovered)}")
print(f"  From FanDuel/web sources: {len(web_recovered)}")
print(f"Total unresolved: {len(unresolved)} / 45")
print(f"  Alias-resolved: {alias_ct}")
print(f"  No data available: {len(unresolved) - alias_ct}")
print(f"Accounted for: {len(all_recovered) + len(unresolved)} / 45")
