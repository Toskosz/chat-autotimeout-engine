"""Hand-labeled chat lines.

Gold labels follow the full-message policy in `src/mock_moderator.py`. Pivot
ids are the lines whose first clause and full text disagree. Those are the
cases the premature-constraint simulation is built to miss.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    id: str
    message: str
    action: str
    violated_rule: str
    why: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "greeting",
        "Pog that last fight, stream is so clean tonight",
        "ALLOW",
        "NONE",
        "Ordinary hype.",
    ),
    Fixture(
        "question",
        "What's the build you used on the boss?",
        "ALLOW",
        "NONE",
        "A question about the game. The word boss is the encounter.",
    ),
    Fixture(
        "gg",
        "gg",
        "ALLOW",
        "NONE",
        "Short and fine.",
    ),
    Fixture(
        "hi",
        "love the stream, can you say hi to my chat?",
        "ALLOW",
        "NONE",
        "A benign request.",
    ),
    Fixture(
        "game_hate",
        "I hate this boss. The mechanics are unfair and the hitboxes are huge.",
        "ALLOW",
        "NONE",
        "Criticism of the encounter. Both the prefix and the full message are clean.",
    ),
    Fixture(
        "trash_mobs",
        "those trash mobs dropped nothing again",
        "ALLOW",
        "NONE",
        "Trash refers to the mobs, and nobody is addressed.",
    ),
    Fixture(
        "retract_insult",
        "You're actually trash. Wait no, that play was insane.",
        "ALLOW",
        "NONE",
        "The insult is withdrawn. The first clause alone still reads as toxicity.",
    ),
    Fixture(
        "redirected_insult",
        "You're trash. The boss, I mean. Not you.",
        "ALLOW",
        "NONE",
        "The insult is redirected at the boss.",
    ),
    Fixture(
        "retract_promo",
        "Go follow my channel right now. Ignore that, I'm here for the subathon.",
        "ALLOW",
        "NONE",
        "The promo is withdrawn in the next sentence.",
    ),
    Fixture(
        "retract_severe",
        "kys... lol jk you're actually cracked at this game",
        "ALLOW",
        "NONE",
        "A self-harm phrase is immediately marked as a joke.",
    ),
    Fixture(
        "late_dox",
        "Nice play! Also here's their home address: 123 Fake Street and phone 555-014-0199.",
        "BAN",
        "DOXXING",
        "The first clause is praise. The violation is the address and the phone number.",
    ),
    Fixture(
        "late_promo",
        "This lag is awful. Also follow my channel for more rants.",
        "TIMEOUT_60S",
        "SELF_PROMO",
        "The first clause is about lag. The promo comes after it.",
    ),
    Fixture(
        "severe",
        "kill yourself",
        "BAN",
        "TOXICITY",
        "An unretracted self-harm directive.",
    ),
    Fixture(
        "directed_insult",
        "you are so bad, uninstall the game",
        "TIMEOUT_60S",
        "TOXICITY",
        "A directed insult with no retraction.",
    ),
    Fixture(
        "hard_promo",
        "CHECK OUT MY CHANNEL BEST CLIPS FOLLOW NOW http://example.com/promo",
        "TIMEOUT_60S",
        "SELF_PROMO",
        "A hard promo, visible in the opening words.",
    ),
    Fixture(
        "code_promo",
        "use code STREAMER at checkout, my brand pays me to say this",
        "TIMEOUT_60S",
        "SELF_PROMO",
        "An affiliate pitch.",
    ),
    Fixture(
        "link_spam",
        "http://example.com http://example.com http://example.com http://example.com",
        "DELETE",
        "SPAM",
        "The same link repeated.",
    ),
    Fixture(
        "word_spam",
        "spam spam spam spam spam spam",
        "DELETE",
        "SPAM",
        "One token repeated.",
    ),
    Fixture(
        "real_name",
        "their real name is Sam and they live at 500 Market Street",
        "BAN",
        "DOXXING",
        "Identity and a street address, in the opening clause.",
    ),
    Fixture(
        "email_dox",
        "email the other mod at jane.doe@example.com and tell them to unban me",
        "BAN",
        "DOXXING",
        "Someone else's email, posted in public chat.",
    ),
)

PIVOT_IDS = frozenset(
    {
        "retract_insult",
        "redirected_insult",
        "retract_promo",
        "retract_severe",
        "late_dox",
        "late_promo",
    }
)

FIXTURES_BY_ID = {fixture.id: fixture for fixture in FIXTURES}
