"""
Corpus künyesi — TÜM scriptler röportaj bilgisini yalnızca buradan okur.
Tarihler ve röportajcılar yayın kaynaklarından doğrulanmıştır (Ekim 2026).

profile      : kitaptaki ana sınıflama
profile_alt  : duyarlılık analizi (yalnızca Oval Office farklı: neutral)
"""

CORPUS = [
    # slug,                 label,                     outlet,                          interviewer,                                   date,         format,               profile,       profile_alt
    ("bunker_inspection",    "Bunker Inspection",       "Fox News Radio",                "Brian Kilmeade",                              "2020-06-03", "Radio",              "supportive",  "supportive"),
    ("chris_wallace_july19", "Chris Wallace",           "Fox News Sunday",               "Chris Wallace",                               "2020-07-19", "Broadcast TV",       "adversarial", "adversarial"),
    ("person_woman_man",     "Person Woman Man",        "Fox News",                      "Marc Siegel",                                 "2020-07-22", "Cable TV",           "supportive",  "supportive"),
    ("axios_hbo",            "Axios/HBO",               "Axios on HBO",                  "Jonathan Swan",                               "2020-08-03", "Cable TV",           "adversarial", "adversarial"),
    ("judge_jeanine_sep12",  "Judge Jeanine",           "Fox News",                      "Jeanine Pirro",                               "2020-09-12", "Cable TV",           "supportive",  "supportive"),
    ("roe_v_wade",           "Roe v. Wade / SCOTUS",    "Fox News (Fox & Friends)",      "Pete Hegseth",                                "2020-09-27", "Cable TV",           "supportive",  "supportive"),
    ("hannity_debate_covid", "Hannity/COVID",           "Fox News",                      "Sean Hannity",                                "2020-10-01", "Cable TV",           "supportive",  "supportive"),
    ("hannity_oct8",         "Hannity Oct. 8",          "Fox News",                      "Sean Hannity",                                "2020-10-08", "Cable TV",           "supportive",  "supportive"),
    ("60_minutes_unedited",  "60 Minutes Unedited",     "CBS",                           "Lesley Stahl",                                "2020-10-22", "Broadcast TV",       "adversarial", "adversarial"),
    ("fox_friends_dec2021",  "Fox & Friends 2021",      "Fox News",                      "Brian Kilmeade, Steve Doocy, Ainsley Earhardt", "2021-12-02", "Cable TV",         "supportive",  "supportive"),
    ("tucker_x_debate",      "Tucker on X",             "X (Twitter)",                   "Tucker Carlson",                              "2023-08-23", "Online video",       "supportive",  "supportive"),
    ("meet_the_press",       "Meet the Press",          "NBC",                           "Kristen Welker",                              "2023-09-17", "Broadcast TV",       "neutral",     "neutral"),
    ("will_cain_show",       "Fox & Friends Weekend",   "Fox News",                      "Will Cain, Pete Hegseth, Rachel Campos-Duffy", "2024-06-02", "Cable TV",          "supportive",  "supportive"),
    ("elon_musk_interview",  "Elon Musk",               "X Spaces",                      "Elon Musk",                                   "2024-08-12", "Online audio",       "supportive",  "supportive"),
    ("oval_office",          "Oval Office (ABC)",       "ABC News",                      "Terry Moran",                                 "2025-04-29", "Broadcast TV",       "adversarial", "neutral"),
    ("60_minutes_2025",      "60 Minutes 2025",         "CBS",                           "Norah O'Donnell",                             "2025-11-02", "Broadcast TV",       "neutral",     "neutral"),
]

# Kapsam dışı (gerekçe: tek başına Trump kelimelerinin yaklaşık dörtte birini oluşturur)
EXCLUDED = {"joe_rogan": "length (25,653 Trump words, more than a quarter of the remaining 16-interview corpus)"}

FIELDS = ["slug", "label", "outlet", "interviewer", "date", "format", "profile", "profile_alt"]
META = {row[0]: dict(zip(FIELDS, row)) for row in CORPUS}
SLUGS = [row[0] for row in CORPUS]
LABEL = {s: META[s]["label"] for s in SLUGS}
DATES = {s: META[s]["date"] for s in SLUGS}
PROFILE = {s: META[s]["profile"] for s in SLUGS}
PROFILE_ALT = {s: META[s]["profile_alt"] for s in SLUGS}


def period(date_str):
    y = int(date_str[:4])
    if y <= 2020:
        return "2020"
    if y == 2021:
        return "2021"
    if y == 2023:
        return "2023"
    if y == 2024:
        return "2024"
    return "2025"
