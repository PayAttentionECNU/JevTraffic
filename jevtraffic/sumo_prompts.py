"""States and typed questions for the SUMO single-intersection controller (v2).

Junctions have 2-4 phases with arbitrary geometry, so nothing is hard-coded: movements are named by approach
(where traffic comes from) and turn, and every phase option is spelled out as the movements it serves. Options
are shuffled during training, so the model must read the option text, not the letter.

Questions:
  * phase (every decision, choose_next_phase mode): which of the currently allowed phases gets the next green;
  * switch (every decision, next_or_not mode, as UniTSA): keep the current green or switch to the next phase;
  * situation (only when the context agent asks): normal / blockage / ambulance / police / fire engine,
    answered from the current state plus a short history table.
"""
from __future__ import annotations

import re

from .decider import Choice
from .sumo_env import COMPASS, SITUATIONS, SumoObs

SYSTEM_PROMPT = "You are a traffic signal control agent. You make fast, decisive signal timing decisions."
STEP_SECONDS = 5

SITUATION_TEXT = {
    "normal": "normal traffic",
    "blockage": "a blocked lane (accident or breakdown): vehicles queue but do not move even on green",
    "ambulance": "an ambulance is approaching and needs priority",
    "police": "a police car is approaching and needs priority",
    "fire_engine": "a fire engine is approaching and needs priority",
}
SITUATION_QUESTION = Choice("What is the current situation at this intersection?",
                            [SITUATION_TEXT[s] for s in SITUATIONS], "situation")


def movement_name(m) -> str:
    return f"{m.approach} {m.turn}"


def phase_name(obs: SumoObs, p: int) -> str:
    ms = [movement_name(obs.movements[i]) for i in obs.phase_movements[p]]
    return " + ".join(ms) if ms else f"phase {p}"


def phase_question(obs: SumoObs, phases: list[int] | None = None) -> tuple[Choice, list[int]]:
    """Choice over the allowed phases (option k <-> phases[k])."""
    phases = list(obs.allowed if phases is None else phases)
    return Choice("Which phase should get the next green?", [phase_name(obs, p) for p in phases], "phase"), phases


def render_state(obs: SumoObs, blocked: set[str] | frozenset = frozenset(), notes: list[str] = ()) -> str:
    """Compact typed state. Vehicles halted on `blocked` lanes (event plan) are not counted in the queue."""
    held = int(obs.green_seconds) if obs.green_seconds is not None else obs.green_steps * STEP_SECONDS
    lines = [f"Intersection with {obs.num_phases} signal phases, time {obs.time:.0f} s.",
             f"Current green: {phase_name(obs, obs.current_phase)} (for {held} s).",
             "Incoming movements (stopped vehicles, moving vehicles, mean wait s, stopped vehicles on the exit):"]
    for i, m in enumerate(obs.movements):
        tag = " [always allowed]" if not m.phases else ""
        bl = [l for l in m.lanes if l in blocked]
        if bl and len(bl) == len(m.lanes):
            lines.append(f"- {movement_name(m)}{tag}: BLOCKED (queue not discharging, excluded)")
            continue
        queue = max(0.0, m.queue - sum(obs.lane_halting.get(l, 0) for l in bl))
        extra = f" ({len(bl)} lane blocked, excluded)" if bl else ""
        lines.append(f"- {movement_name(m)}{tag}: stopped {queue:.0f}{extra}, moving {m.moving:.0f}, "
                     f"wait {m.wait:.0f}, exit stopped {m.down_queue:.0f}")
    for s in obs.special:
        m = obs.movements[s["movement"]]
        lines.append(f"V2X: {s['kind'].replace('_', ' ')} on {movement_name(m)}, {s['dist']:.0f} m from the stop "
                     f"line, speed {s['speed']:.1f} m/s.")
    lines += [f"Note: {n}" for n in notes]
    return "\n".join(lines)


def history_table(obs: SumoObs, history: list) -> str:
    """Short-term memory as a compact table: one row per past decision (oldest first) + now.
    history: list of Snapshot (memory.py)."""
    short = [f"{m.approach[0].upper()}{m.approach[1:3]}-{m.turn[0].upper()}" for m in obs.movements]
    lines = [f"Last {len(history)} decisions (every {STEP_SECONDS} s), per movement stopped/left the intersection; "
             f"columns: {', '.join(f'{s}={movement_name(m)}' for s, m in zip(short, obs.movements))}"]
    for h in history:
        cells = " ".join(f"{s} {q:.0f}/{d:.0f}" for s, q, d in zip(short, h.queues, h.discharged))
        lines.append(f"t={h.time:.0f} green={phase_name(obs, h.phase)} | {cells}")
    return "\n".join(lines)


def situation_state(obs: SumoObs, history: list) -> str:
    return render_state(obs) + "\n" + history_table(obs, history)


# ---------------------------------------------------------------- keep / switch (next_or_not, v3)
KEEP, SWITCH = 0, 1  # option index; the env action is 1 = keep, 0 = switch (tshub next_or_not)


def next_phase(obs: SumoObs) -> int:
    return (obs.current_phase + 1) % obs.num_phases


def switch_question(obs: SumoObs) -> Choice:
    return Choice("Keep the current green or switch to the next phase?",
                  [f"keep: {phase_name(obs, obs.current_phase)}",
                   f"switch to: {phase_name(obs, next_phase(obs))}"], "switch")


def option_to_action(option: int) -> int:
    return 1 if option == KEEP else 0


def action_to_option(action: int) -> int:
    return KEEP if int(action) == 1 else SWITCH


def render_compact(obs: SumoObs, blocked: set[str] | frozenset = frozenset(), notes: list[str] = ()) -> str:
    """Short state for small / fast models: one line per movement, numbers only, phases as indices.
    Roughly a third of render_state's tokens; the question still spells out the two phases in words."""
    held = int(obs.green_seconds) if obs.green_seconds is not None else obs.green_steps * STEP_SECONDS
    lines = [f"t={obs.time:.0f}s green=P{obs.current_phase} for {held}s, next=P{next_phase(obs)}",
             "movement phases: stopped moving wait exit_stopped"]
    for m in obs.movements:
        bl = [l for l in m.lanes if l in blocked]
        ph = "".join(f"P{p}" for p in m.phases) or "free"
        if bl and len(bl) == len(m.lanes):
            lines.append(f"{movement_name(m)} {ph}: BLOCKED")
            continue
        queue = max(0.0, m.queue - sum(obs.lane_halting.get(l, 0) for l in bl))
        lines.append(f"{movement_name(m)} {ph}: {queue:.0f} {m.moving:.0f} {m.wait:.0f} {m.down_queue:.0f}"
                     + (" (blocked lane excluded)" if bl else ""))
    for sv in obs.special:
        lines.append(f"V2X {sv['kind']} on {movement_name(obs.movements[sv['movement']])} {sv['dist']:.0f}m")
    lines += [f"Note: {n}" for n in notes]
    return "\n".join(lines)


STATE_FORMATS = {"full": render_state, "compact": render_compact}


# ---------------------------------------------------------------- layout augmentation (v6)
_COMPASS_RE = re.compile(r"\b(" + "|".join(COMPASS) + r")\b")


def augment_switch(state: str, options: list[str], rng) -> tuple[str, list[str]]:
    """Label-preserving layout augmentation of a keep/switch record: rotate every approach name by the same
    random quarter turn (state and options alike) and shuffle the order of the movement lines. The keep/switch
    decision depends on which movements each phase serves and their queues, not on compass names or the listing
    order, so the teacher's label stays valid; the student cannot key on the layout of the training junctions."""
    k = rng.randrange(4)
    rot = {c: COMPASS[(i + k) % 4] for i, c in enumerate(COMPASS)}

    def turn(text):
        return _COMPASS_RE.sub(lambda m: rot[m.group(1)], text) if k else text

    lines = state.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("- ")]
    moved = [lines[i] for i in idx]
    rng.shuffle(moved)
    for i, l in zip(idx, moved):
        lines[i] = l
    return turn("\n".join(lines)), [turn(o) for o in options]
