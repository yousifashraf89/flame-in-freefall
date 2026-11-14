"""
Step 3 - AI extraction: turn each investigation's text into structured, comparable attributes.
Run once:   export ANTHROPIC_API_KEY=...   (Windows: set ANTHROPIC_API_KEY=...)
            python extract_attributes.py
Output:     data/investigations_enriched.json   (the app picks it up automatically)
Then HAVE A HUMAN SKIM IT - fix anything wrong before the demo.
Uses Anthropic's API; to use another LLM, replace call_llm().
"""
import json, os, sys, time
import pandas as pd

DATA = "data"
MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")

FIELDS = """{
  "one_line_summary": "plain-English, <= 25 words",
  "fire_safety_pillars": ["prevention" | "detection" | "suppression" | "fundamental"],
  "habitat_relevance": "1-2 sentences: why this matters (or not) for spacecraft, ISS, Moon base, rovers, Mars",
  "fuel": "what burned, e.g. 'PMMA and cotton-fiberglass fabric'",
  "sample_geometry": "shape and size of samples / burners / droplets",
  "test_conditions": "O2, pressure, flow, diluents as stated",
  "what_varied": "variables changed between tests",
  "held_constant": "what stayed the same",
  "measurements": ["sensors / diagnostics used"],
  "key_findings": ["up to 4 findings stated in the text"],
  "modeling_value": "what data it offers for predictive flammability / fire models",
  "equipment_implications": "what it informs for fire safety controls or equipment",
  "earth_applications": "terrestrial benefits mentioned, or ''"
}"""

PROMPT = """You are summarizing a NASA microgravity combustion investigation for a fire-safety dashboard.
Use ONLY the text below. If something is not stated, write "not stated" - never guess numbers.
Return ONLY valid JSON with exactly these fields:
{fields}

INVESTIGATION {id} ({acronym}): {title}
Type: {project_type} | Hardware: {hardware}
OBJECTIVE: {objective}
HYPOTHESIS: {hypothesis}
APPROACH: {approach}
RESEARCH IMPACTS: {research_impacts}
PUBLICATION TITLES: {publication_titles}
"""


def call_llm(prompt: str) -> str:
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=MODEL, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


def parse_json(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def main():
    inv = pd.read_csv(f"{DATA}/investigations.csv").fillna("")
    out_path = f"{DATA}/investigations_enriched.json"
    done = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for _, r in inv.iterrows():
        if r.id in done:
            continue
        prompt = PROMPT.format(fields=FIELDS, **{k: str(v)[:6000] for k, v in r.items()})
        for attempt in range(3):
            try:
                done[r.id] = parse_json(call_llm(prompt))
                print("ok ", r.id, r.acronym)
                break
            except Exception as e:
                print("retry", r.id, e, file=sys.stderr)
                time.sleep(3)
        json.dump(done, open(out_path, "w"), indent=1)   # save after each, so reruns resume
    print(f"{len(done)}/{len(inv)} investigations -> {out_path}")


if __name__ == "__main__":
    main()
