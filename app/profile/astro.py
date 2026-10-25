"""Simplified astrology attributes.

Deliberately a stub: tropical (Western) sun sign from date of birth. MyNaksh's real engine is
likely sidereal/Vedic (moon sign, nakshatra, ascendant — which need time and place of birth
and an ephemeris); this module is the seam where that engine would plug in.
"""

from dataclasses import dataclass
from datetime import date

# (month, first day of sign) — a date belongs to the last cutoff it is on/after.
_CUTOFFS = [
    (1, 20, "Aquarius"),
    (2, 19, "Pisces"),
    (3, 21, "Aries"),
    (4, 20, "Taurus"),
    (5, 21, "Gemini"),
    (6, 21, "Cancer"),
    (7, 23, "Leo"),
    (8, 23, "Virgo"),
    (9, 23, "Libra"),
    (10, 23, "Scorpio"),
    (11, 22, "Sagittarius"),
    (12, 22, "Capricorn"),
]


@dataclass(frozen=True)
class SignInfo:
    element: str
    ruler: str
    traits: str


SIGN_INFO: dict[str, SignInfo] = {
    "Aries": SignInfo("Fire", "Mars", "driven, bold, quick to start"),
    "Taurus": SignInfo("Earth", "Venus", "steady, patient, values security"),
    "Gemini": SignInfo("Air", "Mercury", "curious, communicative, adaptable"),
    "Cancer": SignInfo("Water", "Moon", "nurturing, intuitive, protective"),
    "Leo": SignInfo("Fire", "Sun", "confident, expressive, natural leader"),
    "Virgo": SignInfo("Earth", "Mercury", "analytical, practical, detail-oriented"),
    "Libra": SignInfo("Air", "Venus", "diplomatic, fair-minded, relationship-focused"),
    "Scorpio": SignInfo("Water", "Mars/Pluto", "intense, determined, strategic"),
    "Sagittarius": SignInfo("Fire", "Jupiter", "optimistic, freedom-loving, exploratory"),
    "Capricorn": SignInfo("Earth", "Saturn", "disciplined, ambitious, long-term planner"),
    "Aquarius": SignInfo("Air", "Saturn/Uranus", "independent, inventive, future-minded"),
    "Pisces": SignInfo("Water", "Jupiter/Neptune", "empathetic, imaginative, intuitive"),
}


def sun_sign(dob: date) -> str:
    sign = "Capricorn"  # Jan 1–19 falls through to the previous year's Capricorn
    for month, day, name in _CUTOFFS:
        if (dob.month, dob.day) >= (month, day):
            sign = name
    return sign
