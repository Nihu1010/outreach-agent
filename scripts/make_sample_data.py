"""Builds the synthetic demo data. Every person, org and note here is invented.

Outputs:
  data/contacts_sample.csv  - what a user would upload (no answer labels)
  data/eval_labels.csv      - answer key for Phase 5 evals (never shown in the app)
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# 6 personas x 5 contacts. Fields: name, role, org, seniority, email, call_notes, expected_pain_points
# Empty call_notes = no call yet = first touch.
CONTACTS = [
    # Persona 1: Fab manager, worried about leaks and water damage to tools
    ("Dana Whitlock", "Fab Manager", "Northbridge Semiconductor", "senior", "", ""),
    ("Marco Ibarra", "Fab Operations Manager", "Lakeview Microsystems", "senior", "",
     "Marco said a chilled-water line leaked over a weekend last spring and nobody noticed until Monday. "
     "Two lithography tools were down for nine days. He now walks the sub-fab himself every Friday because "
     "the building alarms only cover the main floor.",
     "undetected weekend leaks; nine days of tool downtime; manual Friday sub-fab checks"),
    ("Priya Natarajan", "Fab Manager", "Clearwater Photonics", "senior", "", ""),
    ("Tom Reinholt", "Director of Fab Operations", "Granite Ridge Devices", "senior", "",
     "Tom manages two fabs. His biggest worry is water from roof drains and HVAC condensate reaching tools. "
     "Insurance covered only part of the last incident. He wants earlier warnings, not more reports.",
     "water from roof drains and condensate reaching tools; partial insurance coverage; wants early warnings"),
    ("Lena Ostrowski", "Fab Manager", "Lakeview Microsystems", "senior", "", ""),

    # Persona 2: Cleanroom facilities engineer, particle counts and HVAC
    ("Kevin Mbeki", "Cleanroom Facilities Engineer", "Northbridge Semiconductor", "mid", "",
     "Kevin checks particle counter logs every morning by exporting spreadsheets from three systems. "
     "Excursions are often found hours after they start. He said HVAC filter changes cause most spikes.",
     "manual log exports from three systems; excursions found hours late; HVAC filter changes cause spikes"),
    ("Sofia Alvarez", "Facilities Engineer", "Bayside Biologics", "mid", "", ""),
    ("Raj Kulkarni", "Senior Facilities Engineer", "Clearwater Photonics", "senior", "",
     "Raj said the particle counters work fine but nobody connects counts to what was happening in the room. "
     "Root-cause reviews take a full week because data sits in separate tools.",
     "particle counts not linked to room events; week-long root-cause reviews; data in separate tools"),
    ("Hannah Cole", "Facilities Engineer", "Summit Wafer Labs", "mid", "", ""),
    ("Omar Haddad", "Cleanroom Engineer", "Granite Ridge Devices", "mid", "", ""),

    # Persona 3: University shared-cleanroom lab manager, budget and many users
    ("Dr. Elise Fournier", "Cleanroom Lab Manager", "Westfield University Nanofab", "senior", "",
     "Elise runs a shared cleanroom with over 200 student users. Students forget gowning steps and contamination "
     "is hard to trace to a person or time. Her monitoring budget was cut this year.",
     "200+ student users break gowning protocol; contamination hard to trace; reduced monitoring budget"),
    ("Dr. Sam Okafor", "Director, Microfabrication Facility", "Eastgate Institute of Technology", "senior", "", ""),
    ("Julia Brenner", "Lab Manager", "Westfield University Nanofab", "mid", "", ""),
    ("Dr. Arun Mehta", "Facility Manager", "Riverbend University", "senior", "",
     "Arun said ok.",
     ""),  # thin notes on purpose: the app should ask for more detail, not invent pain points
    ("Claire Dubois", "Cleanroom Manager", "Northern Plains University", "senior", "", ""),

    # Persona 4: Quality / compliance lead, audits and ISO reports
    ("Michael Tanaka", "Quality Assurance Manager", "Bayside Biologics", "senior", "",
     "Michael spends about two days before every audit pulling environmental monitoring records into one report. "
     "Missing data points have led to audit findings twice. He wants a report he can trust without rebuilding it.",
     "two days of manual audit prep; missing data caused two audit findings; needs trustworthy reports"),
    ("Grace Liu", "Compliance Lead", "Summit Wafer Labs", "senior", "", ""),
    ("Victor Santos", "QA Specialist", "Meridian Medical Devices", "mid", "",
     "Victor said deviation reports are written by hand after each excursion. The cleanroom team and QA argue "
     "about when an excursion actually started because timestamps differ between systems.",
     "handwritten deviation reports; disputes over excursion start time; mismatched timestamps"),
    ("Nadia Petrova", "Quality Director", "Meridian Medical Devices", "senior", "", ""),
    ("Ben Archer", "Compliance Manager", "Clearwater Photonics", "senior", "", ""),

    # Persona 5: Equipment maintenance tech, unplanned downtime
    ("Luis Romero", "Equipment Maintenance Technician", "Northbridge Semiconductor", "mid", "",
     "Luis said most of his week is reacting to tool alarms. He has no warning before a pump or chiller fails, "
     "and spare parts take ten days to arrive.",
     "reactive maintenance; no warning before pump or chiller failure; ten-day spare part lead times"),
    ("Aisha Bello", "Maintenance Supervisor", "Lakeview Microsystems", "senior", "", ""),
    ("Derek Lindqvist", "Equipment Engineer", "Summit Wafer Labs", "mid", "",
     "Derek tracks tool health in a personal spreadsheet. When he is on vacation nobody else checks it. "
     "Last year a vacuum pump failure scrapped a full lot of wafers.",
     "tool health tracked in a personal spreadsheet; single point of failure; pump failure scrapped a wafer lot"),
    ("Mei Chen", "Equipment Technician", "Granite Ridge Devices", "mid", "", ""),
    ("Jonas Weber", "Maintenance Manager", "Eastgate Institute of Technology", "senior", "", ""),

    # Persona 6: Junior staff (tests the junior-contact warning)
    ("Ethan Park", "Research Assistant", "Westfield University Nanofab", "junior", "",
     "Ethan said he logs particle readings on paper during his shift and types them in later. "
     "He is not sure who reviews them.",
     "paper logging then retyping; unclear who reviews readings"),
    ("Olivia Grant", "Process Engineering Intern", "Northbridge Semiconductor", "junior", "", ""),
    ("Ryan Moore", "Lab Technician", "Bayside Biologics", "junior", "", ""),
    ("Zoe Adeyemi", "Graduate Student", "Riverbend University", "junior", "",
     "Zoe said her experiments were ruined twice by contamination she found out about only after results came back.",
     "contamination discovered only after results; experiments ruined twice"),
    ("Liam Foster", "Associate Engineer", "Meridian Medical Devices", "junior", "", ""),
]


# First-touch contacts (no call notes) sit at real companies so web research has something to find.
# The people are invented. Contacts WITH call notes stay at invented orgs so no incident is pinned on a real company.
REAL_ORG = {
    "Northbridge Semiconductor": "GlobalFoundries",
    "Lakeview Microsystems": "onsemi",
    "Clearwater Photonics": "Coherent",
    "Granite Ridge Devices": "Wolfspeed",
    "Bayside Biologics": "Regeneron",
    "Summit Wafer Labs": "SkyWater Technology",
    "Westfield University Nanofab": "Cornell NanoScale Science and Technology Facility",
    "Eastgate Institute of Technology": "Penn State Nanofabrication Laboratory",
    "Riverbend University": "University of Rochester URnano",
    "Northern Plains University": "Minnesota Nano Center",
    "Meridian Medical Devices": "Medtronic",
}

# Optional context for follow-ups: who had the call, and where. Empty = "Thank you again for speaking with me."
MET = {
    "Marco Ibarra": ("Sam, my research partner,", "ALD 2026 in Tampa"),
    "Kevin Mbeki": ("", "the SEMICON West career fair"),
    "Dr. Elise Fournier": ("Sam, my research partner,", ""),
}


def email_for(name: str, org: str) -> str:
    first = name.replace("Dr. ", "").split()[0].lower()
    domain = org.lower().replace(" ", "").replace(",", "")[:18]
    return f"{first}@{domain}.example"  # .example domains are reserved and never deliver


def main() -> None:
    DATA.mkdir(exist_ok=True)
    with open(DATA / "contacts_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contact_id", "name", "role", "org", "seniority", "email", "call_notes", "met_by", "met_at"])
        for i, row in enumerate(CONTACTS, start=1):
            name, role, org, seniority, _, notes = row[:6]
            if not notes.strip():
                org = REAL_ORG.get(org, org)
            met_by, met_at = MET.get(name, ("", ""))
            w.writerow([f"C{i:02d}", name, role, org, seniority, email_for(name, org), notes, met_by, met_at])

    with open(DATA / "eval_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contact_id", "stage", "expected_pain_points"])
        for i, row in enumerate(CONTACTS, start=1):
            notes = row[5]
            expected = row[6] if len(row) > 6 else ""
            stage = "follow_up" if notes.strip() else "first_touch"
            w.writerow([f"C{i:02d}", stage, expected])

    print(f"Wrote {len(CONTACTS)} contacts")


if __name__ == "__main__":
    main()
