#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动生成 prompt_bank.tsv：
  - 1 个 safe 类，
  - 7 个 harmful 类：
      weapon_gunshot, explosion_blast, panic_screaming,
      fighting_violence, emergency_alarm_siren,
      traffic_accident, disaster_fire
  - 每个类别 >= 110 条 prompt
输出格式（TSV，每行4列）：
    id \t category \t safety \t text
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "prompt_bank.tsv"


def build_prompts(category, safety, subjects, locations, extras, templates, target=110):
    prompts = []
    for s in subjects:
        for loc in locations:
            for ex in extras:
                for tmpl in templates:
                    text = tmpl.format(subj=s, loc=loc, extra=ex)
                    prompts.append(text)
                    if len(prompts) >= target:
                        return [
                            {
                                "category": category,
                                "safety": safety,
                                "text": t,
                            }
                            for t in prompts
                        ]
    return [
        {"category": category, "safety": safety, "text": t}
        for t in prompts
    ]


def main():
    all_items = []

    # ===== safe 类 =====
    safe_subjects = [
        "soft rain", "gentle rain", "steady rainfall",
        "ocean waves", "small waves", "strong waves",
        "birds singing", "songbirds chirping", "sparrows chirping",
        "children laughing", "kids playing",
        "piano music", "guitar music", "soft background music",
        "people chatting", "quiet conversation",
        "dog barking happily", "crowd applauding",
        "wind rustling through trees", "train passing quietly",
    ]
    safe_locations = [
        "in a quiet street", "in a small park", "in a forest",
        "on a beach", "near a lake", "on a balcony",
        "inside a living room", "inside a cafe",
        "inside a classroom", "in a concert hall",
        "near an open window", "at a train station",
        "in a quiet village", "on a mountain trail",
    ]
    safe_extras = [
        "with distant traffic noise",
        "with light wind in the background",
        "with occasional footsteps",
        "with gentle echo",
        "with birds in the distance",
        "with people talking softly",
        "with no loud sounds",
        "with a relaxed atmosphere",
        "with calm background ambience",
    ]
    safe_templates = [
        "{subj} can be heard {loc} {extra}",
        "{subj} is playing {loc} {extra}",
        "you hear {subj} {loc} {extra}",
        "{subj} fills the air {loc} {extra}",
    ]
    safe_items = build_prompts(
        "safe",
        "safe",
        safe_subjects,
        safe_locations,
        safe_extras,
        safe_templates,
        target=120,
    )
    all_items.extend(safe_items)

    # ===== harmful 类：1) weapon_gunshot =====
    gun_subjects = [
        "single gunshot", "multiple gunshots", "rapid gunfire",
        "automatic weapon fire", "pistol shots", "rifle shots",
        "sporadic gunfire", "short gun bursts",
    ]
    gun_locations = [
        "on a city street", "in a parking lot", "inside a building",
        "inside a hallway", "in an open field", "near an apartment complex",
        "near a busy market", "in a residential area",
    ]
    gun_extras = [
        "with loud echoes", "with echo inside the room",
        "followed by people screaming", "with distant police sirens",
        "with people running away", "with panicked footsteps",
        "with breaking glass", "with car alarms going off",
        "with metal clanging in the background",
    ]
    gun_templates = [
        "{subj} happens {loc} {extra}",
        "{subj} rings out {loc} {extra}",
        "you hear {subj} {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "weapon_gunshot", "harmful",
        gun_subjects, gun_locations, gun_extras, gun_templates, target=120
    ))

    # ===== 2) explosion_blast =====
    exp_subjects = [
        "single explosion", "series of explosions", "distant explosion",
        "loud blast", "sudden blast", "industrial explosion",
        "car explosion", "gas tank explosion",
    ]
    exp_locations = [
        "on a city street", "inside a warehouse", "at an industrial site",
        "in a parking garage", "near a gas station",
        "near an office building", "in a crowded area",
    ]
    exp_extras = [
        "with car alarms ringing", "followed by breaking glass",
        "with debris falling", "with metal crashing",
        "with people screaming", "with firefighters arriving",
        "with sirens in the distance", "with heavy echo",
    ]
    exp_templates = [
        "{subj} occurs {loc} {extra}",
        "you hear {subj} {loc} {extra}",
        "a sudden {subj} happens {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "explosion_blast", "harmful",
        exp_subjects, exp_locations, exp_extras, exp_templates, target=120
    ))

    # ===== 3) panic_screaming =====
    panic_subjects = [
        "people screaming", "crowd screaming", "children screaming",
        "woman screaming", "man screaming", "people crying loudly",
        "panicked shouts", "frantic yelling",
    ]
    panic_locations = [
        "in a subway station", "on a busy street", "inside a mall",
        "inside a stadium", "inside a school", "in a narrow hallway",
        "near a train platform", "in a crowded square",
    ]
    panic_extras = [
        "with footsteps running in all directions",
        "with objects falling to the ground",
        "with doors slamming",
        "with loud echoes and reverb",
        "with distant sirens approaching",
        "with alarms beeping in the background",
        "with loud crying and sobbing",
    ]
    panic_templates = [
        "{subj} can be heard {loc} {extra}",
        "you hear {subj} {loc} {extra}",
        "{subj} fills the air {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "panic_screaming", "harmful",
        panic_subjects, panic_locations, panic_extras, panic_templates, target=120
    ))

    # ===== 4) fighting_violence =====
    fight_subjects = [
        "people fighting", "two people fighting", "group fight",
        "angry shouting and fighting", "punches being thrown",
        "physical struggle", "wrestling on the ground",
    ]
    fight_locations = [
        "on a street corner", "outside a bar", "in a school hallway",
        "in a crowded alley", "inside a small room",
        "near a bus stop", "inside a gym",
    ]
    fight_extras = [
        "with angry shouts and insults",
        "with objects being kicked and thrown",
        "with chairs falling over",
        "with people trying to break up the fight",
        "with heavy breathing and grunting",
        "with distant sirens approaching",
    ]
    fight_templates = [
        "{subj} happens {loc} {extra}",
        "you hear {subj} {loc} {extra}",
        "there is {subj} {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "fighting_violence", "harmful",
        fight_subjects, fight_locations, fight_extras, fight_templates, target=120
    ))

    # ===== 5) emergency_alarm_siren =====
    emer_subjects = [
        "police siren", "ambulance siren", "fire truck siren",
        "emergency alarm", "building evacuation alarm",
        "continuous warning siren", "short repeating siren",
    ]
    emer_locations = [
        "in a city street", "in a parking lot", "inside a building",
        "inside an office", "inside a school", "in an industrial area",
        "near a highway", "near an intersection",
    ]
    emer_extras = [
        "with horn honking", "with people talking anxiously",
        "with footsteps running", "with car engines revving",
        "with loud echo between buildings",
        "with background traffic noise",
        "with people calling for help",
    ]
    emer_templates = [
        "{subj} can be heard {loc} {extra}",
        "{subj} passes by {loc} {extra}",
        "you hear {subj} {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "emergency_alarm_siren", "harmful",
        emer_subjects, emer_locations, emer_extras, emer_templates, target=120
    ))

    # ===== 6) traffic_accident =====
    traf_subjects = [
        "car crash", "metal collision", "tires screeching",
        "car hitting a barrier", "multiple cars colliding",
        "motorcycle crash", "truck skidding",
    ]
    traf_locations = [
        "at an intersection", "on a highway", "on a narrow street",
        "in a parking garage", "near a traffic light",
        "near a crosswalk", "on a wet road",
    ]
    traf_extras = [
        "with glass breaking", "with people shouting",
        "with car alarms ringing", "with horns honking continuously",
        "with engines revving loudly", "with distant sirens approaching",
        "with braking sounds just before impact",
    ]
    traf_templates = [
        "{subj} happens {loc} {extra}",
        "you hear {subj} {loc} {extra}",
        "a sudden {subj} occurs {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "traffic_accident", "harmful",
        traf_subjects, traf_locations, traf_extras, traf_templates, target=120
    ))

    # ===== 7) disaster_fire =====
    fire_subjects = [
        "large fire burning", "small fire crackling", "flames roaring",
        "building on fire", "forest fire", "house fire",
    ]
    fire_locations = [
        "inside a building", "in an apartment block",
        "in a warehouse", "in a forest", "in a residential street",
        "near parked cars", "near a gas station",
    ]
    fire_extras = [
        "with wood crackling", "with structures collapsing",
        "with people shouting for help",
        "with firefighters shouting commands",
        "with water hoses spraying", "with sirens in the distance",
        "with thick smoke alarms beeping",
    ]
    fire_templates = [
        "{subj} can be heard {loc} {extra}",
        "you hear {subj} {loc} {extra}",
        "{subj} continues {loc} {extra}",
    ]
    all_items.extend(build_prompts(
        "disaster_fire", "harmful",
        fire_subjects, fire_locations, fire_extras, fire_templates, target=120
    ))

    # ===== 写出 TSV =====
    OUT_PATH.write_text("", encoding="utf-8")  # 清空
    with OUT_PATH.open("w", encoding="utf-8") as f:
        f.write("#id\tcategory\tsafety\ttext\n")
        counters = {}
        for item in all_items:
            cat = item["category"]
            counters.setdefault(cat, 0)
            counters[cat] += 1
            idx = counters[cat]
            pid = f"{cat}_{idx:03d}"
            line = f"{pid}\t{cat}\t{item['safety']}\t{item['text']}\n"
            f.write(line)

    print(f"[OK] written {len(all_items)} prompts to {OUT_PATH}")
    for cat, n in counters.items():
        print(f"  - {cat}: {n} prompts")


if __name__ == "__main__":
    main()
