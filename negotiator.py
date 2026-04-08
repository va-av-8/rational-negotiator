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
Maximize the Nash Bargaining Solution: the product of both players' surpluses above BATNA.
    (your_value - your_batna) × (their_value - their_batna)
This is maximized when BOTH players gain significantly above their outside options.
A greedy offer that the opponent rejects is worth ZERO. A fair deal you both accept is worth a lot.

KEY INSIGHT — THE OPPONENT IS RATIONAL:
- The opponent will ACCEPT any offer where their value > their BATNA
- The opponent will REJECT offers that give them too little
- You cannot know their exact valuations, but you can infer:
  items you value least are likely valuable to them too
  → give them items you care less about, keep items you care most about

STRATEGY BY ROUND:
- Round 1: Propose a deal where you get ~65% of YOUR value, opponent gets reasonable share.
  Don't be too greedy — a rejected offer wastes a round.
- Middle rounds: If offers are being rejected, you are likely asking too much.
  Increase opponent's share by ~5-10% of total value each round.
- Last 2 rounds: Accept anything above your BATNA. Time pressure overrides everything.

FINDING A GOOD SPLIT:
1. Identify which items you value most → prioritize keeping those
2. Identify which items you value least → offer those to the opponent generously
3. Check: does your allocation give you significantly above your BATNA?
4. Check: does the opponent's allocation seem reasonable (not nearly zero)?
5. If both checks pass → propose it

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

    # Try to find any JSON object
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


# =============================================================================
# LLM Integration
# =============================================================================

def prepare_context(obs: dict[str, Any]) -> str:
    """Prepare context message for LLM."""
    game_index = obs.get("game_index", 0)
    valuations_self = obs.get("valuations_self", [])
    batna_self = obs.get("batna_self", 0)
    quantities = obs.get("quantities", [])
    round_index = obs.get("round_index", 1)
    max_rounds = obs.get("max_rounds", 5)
    discount = obs.get("discount", 0.98)

    # Calculate max possible value (all items for myself)
    max_possible = calculate_value(quantities, valuations_self)
    rounds_left = max_rounds - round_index
    discounted_batna = batna_self * (discount ** (round_index - 1))

    with _history_lock:
        my_prev_offers = _my_offer_history.get(game_index, [])

    history_text = ""
    if my_prev_offers:
        history_text = "\nMy previous offers to opponent (with my value for each):\n"
        for i, offer in enumerate(my_prev_offers, 1):
            my_alloc = self_from_other(offer, quantities)
            my_val = calculate_value(my_alloc, valuations_self)
            history_text += f"  Offer {i}: gave opponent {offer}, kept {my_alloc} (my value: {my_val})\n"

    context = f"""GAME STATE:
- My valuations per item: {valuations_self}
- Total quantities: {quantities}
- My BATNA: {batna_self} (discounted this round: {discounted_batna:.1f})
- Maximum possible value (all items): {max_possible}
- Round: {round_index} of {max_rounds} ({rounds_left} rounds left)
- Discount per round: {discount}
{history_text}
TARGET: Propose allocation where MY value > {discounted_batna:.1f} (discounted BATNA).
The higher above BATNA, the better. Aim for at least {min(discounted_batna * 1.3, max_possible * 0.7):.0f}.
"""
    return context


def call_llm(context: str) -> dict[str, Any] | None:
    """Call LLM to get allocation proposal."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ]
        )

        response_text = response.choices[0].message.content.strip()
        logger.info(f"LLM response: {response_text}")

        # Parse JSON from response
        if response_text.startswith("{"):
            return json.loads(response_text)

        # Try to extract JSON from text
        match = re.search(r"\{[^}]+\}", response_text)
        if match:
            return json.loads(match.group())

        return None

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

    # TODO: возможно вернуть позже
    # -------------------------------------------------------------------------
    # M3: Cannot give opponent 0 items or keep 0 items for myself
    # -------------------------------------------------------------------------
    # if all(x == 0 for x in allocation_other):
    #     sorted_indices = sorted(range(len(valuations_self)), key=lambda i: valuations_self[i])
    #     for i in sorted_indices:
    #         if quantities[i] > 0 and allocation_self[i] > 0:
    #             allocation_other[i] = 1
    #             allocation_self[i] -= 1
    #             break
    #
    # if all(x == 0 for x in allocation_self):
    #     sorted_indices = sorted(range(len(valuations_self)), key=lambda i: valuations_self[i])
    #     for i in sorted_indices:
    #         if allocation_other[i] > 0:
    #             allocation_other[i] -= 1
    #             allocation_self[i] += 1
    #             break

    # TODO: возможно вернуть позже
    # -------------------------------------------------------------------------
    # M2: My value must be >= BATNA
    # -------------------------------------------------------------------------
    # valuations_self = obs["valuations_self"]
    # batna_self = obs["batna_self"]
    # if calculate_value(allocation_self, valuations_self) < batna_self:
    #     allocation_self, allocation_other = adjust_for_batna(
    #         allocation_self, allocation_other,
    #         quantities, valuations_self, batna_self
    #     )
    #     if allocation_self is None:
    #         return None, None  # -> WALK

    # TODO: возможно вернуть позже
    # -------------------------------------------------------------------------
    # M1: Cannot decrease allocation_other vs previous offer
    # -------------------------------------------------------------------------
    # with _history_lock:
    #     prev_offers = _my_offer_history.get(game_index, [])
    #
    # if prev_offers:
    #     prev_other = prev_offers[-1]
    #     allocation_other = [max(new, prev) for new, prev in zip(allocation_other, prev_other)]
    #     allocation_self = self_from_other(allocation_other, quantities)
    #
    #     # M2 after M1: check BATNA again
    #     if calculate_value(allocation_self, valuations_self) < batna_self:
    #         allocation_self, allocation_other = adjust_for_batna(
    #             allocation_self, allocation_other,
    #             quantities, valuations_self, batna_self
    #         )
    #         if allocation_self is None:
    #             return None, None  # -> WALK

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

        # Apply discount for current round
        round_index = obs.get("round_index", 1)
        discount = obs.get("discount", 0.98)
        discounted_batna = batna_self * (discount ** (round_index - 1))

        if value >= discounted_batna:
            # M5: Don't walk away from offer better than BATNA
            return {"action": "ACCEPT"}
        else:
            # M4: Don't accept offer worse than BATNA
            return {"action": "WALK"}

    # =========================================================================
    # PROPOSE - LLM + constraint enforcement
    # =========================================================================
    context = prepare_context(obs)
    llm_result = call_llm(context)

    if llm_result is None:
        # Fallback on LLM error - use greedy offer
        fallback = make_greedy_offer(quantities, valuations_self, batna_self)
        if fallback is None:
            return {"action": "WALK", "reason": "cannot_satisfy_batna"}
        a_self, a_other = fallback

        # Save to history
        with _history_lock:
            _my_offer_history.setdefault(game_index, []).append(a_other)

        return {"allocation_self": a_self, "allocation_other": a_other}

    a_self, a_other = enforce_constraints(llm_result, obs, game_index)

    if a_self is None:
        return {"action": "WALK", "reason": "constraints_unsatisfiable"}

    # Save to history
    with _history_lock:
        _my_offer_history.setdefault(game_index, []).append(a_other)

    return {"allocation_self": a_self, "allocation_other": a_other}
