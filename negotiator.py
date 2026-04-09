"""
Rational Negotiator - Hybrid LLM + Deterministic Rules

Combines LLM reasoning for proposals with deterministic enforcement
of M1-M5 constraints to avoid common negotiation mistakes.
"""
import json
import logging
import os
import re
import threading
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

logger = logging.getLogger("rational_negotiator")

# =============================================================================
# LLM Client Initialization
# =============================================================================

def init_client() -> tuple[OpenAI, str]:
    """Initialize OpenAI-compatible client with auto-detection of provider."""
    if key := os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        return OpenAI(api_key=key), model
    if key := os.environ.get("OPENROUTER_API_KEY"):
        model = os.environ.get("LLM_MODEL") or "openai/gpt-4o-mini"
        return OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1"
        ), model
    raise RuntimeError(
        "No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY."
    )


client, MODEL = init_client()

# =============================================================================
# Offer History (thread-safe)
# =============================================================================

# game_index -> list of allocation_other I proposed to opponent
_my_offer_history: dict[int, list[list[int]]] = {}
_history_lock = threading.Lock()

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a strategic negotiator dividing items between yourself and an opponent.

GAME STRUCTURE:
- Items have different values for each player (private information)
- Each player has a BATNA — value they get if no deal is reached
- Payoffs decrease each round by the discount factor
- You must propose how to split ALL items

YOUR OBJECTIVE:
Maximize the Nash Bargaining Solution: the product of BOTH players' surpluses above BATNA.
    (your_value - your_batna) × (their_value - their_batna)
This means BOTH players must gain above their outside options.
A greedy offer the opponent rejects is worth ZERO. A fair deal accepted is worth a lot.

KEY INSIGHT — INTEGRATIVE BARGAINING:
The opponent has DIFFERENT private valuations than you.
Items you value LEAST are often items they value MOST.
→ Give them items you care less about. Keep items you care most about.
→ This creates deals where BOTH sides feel they won — maximizing Nash Welfare.

The context will show you item rankings and a suggested Nash target.
Use these as your anchor — don't deviate far from the Nash target.

STRATEGY BY ROUND:
- Round 1: Start near the Nash target. Don't be greedy — rejected offers waste rounds.
- Middle rounds: If rejected, increase opponent's share of low-value items by 1-2 units.
- Last round: Close the deal. Accept almost anything above a very low threshold.

RESPOND with JSON only, no explanation:
{"allocation_self": [int, ...], "allocation_other": [int, ...]}

allocation_self[i] + allocation_other[i] MUST equal quantities[i] exactly.
All values must be non-negative integers.
"""

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_value(allocation: list[int], valuations: list[int]) -> int:
    """Calculate total value of an allocation given valuations."""
    return sum(a * v for a, v in zip(allocation, valuations))


def other_from_self(allocation_self: list[int], quantities: list[int]) -> list[int]:
    """Compute allocation_other from allocation_self."""
    return [q - a for q, a in zip(quantities, allocation_self)]


def self_from_other(allocation_other: list[int], quantities: list[int]) -> list[int]:
    """Compute allocation_self from allocation_other."""
    return [q - a for q, a in zip(quantities, allocation_other)]


def compute_nbs_allocation(
    quantities: list[int],
    valuations_self: list[int],
    batna_self: int
) -> tuple[list[int], list[int]]:
    """
    Approximate Nash Bargaining Solution allocation.

    Strategy: Keep items we value most (per unit), give items we value least.
    This is Pareto-efficient when opponents have complementary valuations.

    Returns (allocation_self, allocation_other) targeting NBS.
    """
    n = len(quantities)
    max_possible = calculate_value(quantities, valuations_self)

    # Target: take 50-55%
    nbs_target = max_possible * 0.5

    # Sort item types by value per unit (descending) — keep most valuable first
    item_priority = sorted(range(n), key=lambda i: valuations_self[i], reverse=True)

    allocation_self = [0] * n
    allocation_other = list(quantities)
    current_value = 0

    # Greedily take items in order of value until we hit NBS target
    for i in item_priority:
        if current_value >= nbs_target:
            break
        needed_value = nbs_target - current_value
        units_to_take = min(
            quantities[i],
            # Take only as many units as needed to reach target
            int(needed_value / valuations_self[i]) + 1 if valuations_self[i] > 0 else quantities[i]
        )
        units_to_take = max(0, min(units_to_take, quantities[i]))
        allocation_self[i] = units_to_take
        allocation_other[i] = quantities[i] - units_to_take
        current_value += units_to_take * valuations_self[i]

    return allocation_self, allocation_other


def make_greedy_offer(
    quantities: list[int],
    valuations_self: list[int],
    batna_self: int
) -> tuple[list[int], list[int]] | None:
    """
    Create maximally greedy offer where my value >= batna_self and opponent gets >= 1 item.
    Used ONLY as fallback when LLM fails.

    Algorithm:
    1. Start with everything for myself (M2 satisfied, M3 violated)
    2. Give opponent 1 unit of my least valuable item (M3)
    3. If my value < batna_self - impossible to satisfy both -> None -> WALK

    Returns:
        (allocation_self, allocation_other) or None if impossible
    """
    allocation_self = list(quantities)
    allocation_other = [0] * len(quantities)

    # M3: give 1 unit of least valuable item to opponent
    sorted_indices = sorted(range(len(valuations_self)), key=lambda i: valuations_self[i])
    for i in sorted_indices:
        if quantities[i] > 0:
            allocation_other[i] = 1
            allocation_self[i] -= 1
            break

    # M2: check if my value >= batna_self
    if calculate_value(allocation_self, valuations_self) < batna_self:
        return None  # cannot satisfy M2 and M3 simultaneously -> WALK

    return allocation_self, allocation_other


def adjust_for_batna(
    allocation_self: list[int],
    allocation_other: list[int],
    quantities: list[int],
    valuations_self: list[int],
    batna_self: int
) -> tuple[list[int] | None, list[int] | None]:
    """
    Minimally adjust offer so that my value >= batna_self (M2).
    Takes items from opponent starting with most valuable for us.
    Returns (None, None) if M2 cannot be satisfied -> WALK.
    """
    allocation_self = list(allocation_self)
    allocation_other = list(allocation_other)

    while calculate_value(allocation_self, valuations_self) < batna_self:
        taken = False
        # Take most valuable items first
        for i in sorted(range(len(valuations_self)),
                        key=lambda i: valuations_self[i], reverse=True):
            if allocation_other[i] > 0:
                allocation_other[i] -= 1
                allocation_self[i] += 1
                taken = True
                break
        if not taken:
            return None, None

    return allocation_self, allocation_other


# =============================================================================
# Observation Parsing
# =============================================================================

def parse_observation(message_text: str) -> dict[str, Any]:
    """Extract JSON observation from message text."""
    # Try to find JSON block in markdown
    patterns = [
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
        r"Observation:\s*(\{.*?\})",
    ]

    for pattern in patterns:
        match = re.search(pattern, message_text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try to find any JSON object using depth tracking
    brace_start = message_text.find("{")
    if brace_start != -1:
        depth = 0
        for i, c in enumerate(message_text[brace_start:], brace_start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(message_text[brace_start:i+1])
                    except json.JSONDecodeError:
                        break

    return {}


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Robustly extract JSON object from LLM response text.
    Uses depth-tracking to handle nested structures (arrays inside objects).
    Fixes the broken regex approach that failed on nested JSON.
    """
    # First try: direct parse if text is pure JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: find JSON object using brace depth tracking
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    for i, c in enumerate(text[brace_start:], brace_start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start:i+1])
                except json.JSONDecodeError:
                    break

    return None


# =============================================================================
# LLM Integration
# =============================================================================

def prepare_context(obs: dict[str, Any]) -> str:
    """Prepare context message for LLM with Nash Bargaining guidance."""
    game_index = obs.get("game_index", 0)
    valuations_self = obs.get("valuations_self", [])
    batna_self = obs.get("batna_self", 0)
    quantities = obs.get("quantities", [])
    round_index = obs.get("round_index", 1)
    max_rounds = obs.get("max_rounds", 5)
    discount = obs.get("discount", 0.98)

    # Calculate key values
    max_possible = calculate_value(quantities, valuations_self)
    rounds_left = max_rounds - round_index
    discounted_batna = batna_self * (discount ** (round_index - 1))

    # Nash Bargaining Solution approximation
    nbs_alloc_self, nbs_alloc_other = compute_nbs_allocation(
        quantities, valuations_self, batna_self
    )
    nbs_my_value = calculate_value(nbs_alloc_self, valuations_self)
    nbs_target = max_possible * 0.5

    # Item ranking by value per unit
    item_ranking = sorted(range(len(valuations_self)),
                          key=lambda i: valuations_self[i], reverse=True)
    ranking_text = "Item priority (keep → give away):\n"
    for rank, i in enumerate(item_ranking):
        label = "KEEP (high value)" if rank < len(item_ranking) // 2 + 1 else "GIVE (low value)"
        ranking_text += f"  Item {i}: value={valuations_self[i]}/unit, qty={quantities[i]} → {label}\n"

    # History
    with _history_lock:
        my_prev_offers = _my_offer_history.get(game_index, [])

    history_text = ""
    if my_prev_offers:
        history_text = "\nMy previous offers (what I gave opponent):\n"
        for i, offer in enumerate(my_prev_offers, 1):
            my_alloc = self_from_other(offer, quantities)
            my_val = calculate_value(my_alloc, valuations_self)
            history_text += f"  Offer {i}: gave opponent {offer}, kept {my_alloc} (my value: {my_val})\n"

    context = f"""GAME STATE:
- My valuations per item: {valuations_self}
- Total quantities: {quantities}
- My BATNA: {batna_self} (discounted this round: {discounted_batna:.1f})
- My maximum possible value (all items): {max_possible}
- Round: {round_index} of {max_rounds} ({rounds_left} rounds left)
- Discount per round: {discount}

{ranking_text}
NASH BARGAINING TARGET:
- Suggested allocation for me: {nbs_alloc_self} (gives me ~{nbs_my_value:.0f})
- Suggested allocation for opponent: {nbs_alloc_other}
- This targets Nash value of ~{nbs_target:.0f} for me (halfway between BATNA and max)
- My surplus above BATNA at this allocation: {nbs_my_value - batna_self:.0f}
{history_text}
INSTRUCTION: Start near the Nash target above. 
Keep items you value most. Give opponent items you value least — they likely value them more.
Aim for my value ≈ {nbs_target:.0f}. Going much higher risks rejection and wastes rounds.
"""
    return context


def call_llm(context: str) -> dict[str, Any] | None:
    """Call LLM to get allocation proposal."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            temperature=0.2,  # Low temperature for consistent Nash-seeking behavior
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ]
        )

        response_text = response.choices[0].message.content.strip()
        logger.info(f"LLM response: {response_text}")

        # Use robust depth-tracking JSON extractor (fixes broken regex)
        return extract_json_from_text(response_text)

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


# =============================================================================
# Constraint Enforcement (M1-M5)
# =============================================================================

def enforce_constraints(
    llm_result: dict[str, Any],
    obs: dict[str, Any],
    game_index: int
) -> tuple[list[int] | None, list[int] | None]:
    """
    Validate LLM proposal - only check allocation sums.

    Returns:
        (allocation_self, allocation_other) or (None, None) if invalid -> WALK
    """
    quantities = obs["quantities"]

    # Extract allocations from LLM result
    allocation_self = llm_result.get("allocation_self")
    allocation_other = llm_result.get("allocation_other")

    if allocation_self is None:
        return None, None

    # Ensure integers
    allocation_self = [int(x) for x in allocation_self]

    if allocation_other is None:
        allocation_other = other_from_self(allocation_self, quantities)
    else:
        allocation_other = [int(x) for x in allocation_other]

    # Validate sum equals quantities - if not, WALK
    for i in range(len(quantities)):
        if allocation_self[i] + allocation_other[i] != quantities[i]:
            return None, None  # -> WALK

    # Validate non-negative values
    for i in range(len(quantities)):
        if allocation_self[i] < 0 or allocation_other[i] < 0:
            return None, None  # -> WALK

    return allocation_self, allocation_other


# =============================================================================
# Main Handler
# =============================================================================

def handle_negotiation_message(message_text: str) -> dict[str, Any]:
    """
    Process negotiation message and return response.

    For ACCEPT_OR_REJECT: purely deterministic (no LLM)
    For PROPOSE: LLM + enforce_constraints
    """
    obs = parse_observation(message_text)

    if not obs:
        logger.warning("Could not parse observation from message")
        return {"action": "WALK", "reason": "parse_error"}

    quantities = obs.get("quantities", [])
    valuations_self = obs.get("valuations_self", [])
    batna_self = obs.get("batna_self", 0)
    game_index = obs.get("game_index", 0)
    action = obs.get("action", "").upper()
    pending = obs.get("pending_offer")

    # =========================================================================
    # ACCEPT_OR_REJECT - Purely deterministic, no LLM needed
    # =========================================================================
    if action == "ACCEPT_OR_REJECT":
        if pending is None:
            return {"action": "WALK", "reason": "no_pending_offer"}

        offer_allocation_self = pending.get("offer_allocation_self", [])
        value = calculate_value(offer_allocation_self, valuations_self)

        round_index = obs.get("round_index", 1)
        max_rounds = obs.get("max_rounds", 10)
        discount = obs.get("discount", 0.98)
        discounted_batna = batna_self * (discount ** (round_index - 1))

        # В первой трети — держим полный порог
        # В середине — снижаем до 70%
        # В последней трети — принимаем почти всё (30%)
        progress = round_index / max_rounds
        if progress < 0.33:
            threshold = discounted_batna
        elif progress < 0.67:
            threshold = discounted_batna * 0.7
        else:
            threshold = discounted_batna * 0.3

        if value >= threshold:
            return {"action": "ACCEPT"}
        else:
            return {"action": "WALK"}

    # =========================================================================
    # PROPOSE - LLM + constraint enforcement
    # =========================================================================
    context = prepare_context(obs)
    llm_result = call_llm(context)

    if llm_result is None:
        # Fallback on LLM error - use NBS allocation instead of greedy
        nbs_self, nbs_other = compute_nbs_allocation(quantities, valuations_self, batna_self)
        if calculate_value(nbs_self, valuations_self) >= batna_self:
            with _history_lock:
                _my_offer_history.setdefault(game_index, []).append(nbs_other)
            return {"allocation_self": nbs_self, "allocation_other": nbs_other}

        fallback = make_greedy_offer(quantities, valuations_self, batna_self)
        if fallback is None:
            return {"action": "WALK", "reason": "cannot_satisfy_batna"}
        a_self, a_other = fallback

        with _history_lock:
            _my_offer_history.setdefault(game_index, []).append(a_other)

        return {"allocation_self": a_self, "allocation_other": a_other}

    a_self, a_other = enforce_constraints(llm_result, obs, game_index)

    if a_self is None:
        # LLM returned invalid allocation — use NBS as fallback instead of WALK
        nbs_self, nbs_other = compute_nbs_allocation(quantities, valuations_self, batna_self)
        if calculate_value(nbs_self, valuations_self) >= batna_self:
            with _history_lock:
                _my_offer_history.setdefault(game_index, []).append(nbs_other)
            return {"allocation_self": nbs_self, "allocation_other": nbs_other}
        return {"action": "WALK", "reason": "constraints_unsatisfiable"}

    # Save to history
    with _history_lock:
        _my_offer_history.setdefault(game_index, []).append(a_other)

    return {"allocation_self": a_self, "allocation_other": a_other}
