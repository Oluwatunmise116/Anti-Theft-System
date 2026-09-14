"""
Phone-number normalisation for the SMS OTP fallback.

A holder's saved number is reduced to E.164 ("+2348031234567") with the
phonenumbers library before anything is sent, so "0803 123 4567",
"08031234567" and "+234 803 123 4567" are all the same destination. A
number that phonenumbers does not consider valid normalises to None and is
never sent to.
"""
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat


def normalize_phone(raw, default_region: str = "NG"):
    """A registered holder's number, as typed by staff. National formats are
    interpreted in `default_region`. Returns E.164 or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        number = phonenumbers.parse(text, default_region)
    except NumberParseException:
        return None
    if not phonenumbers.is_valid_number(number):
        return None
    return phonenumbers.format_number(number, PhoneNumberFormat.E164)


def mask_phone(e164) -> str:
    """'+2348031234567' -> '+234 ••• ••• 67'. For display only."""
    if not e164:
        return ""
    try:
        number = phonenumbers.parse(e164, None)
        country = f"+{number.country_code}"
    except NumberParseException:
        country = e164[:4]
    return f"{country} ••• ••• {e164[-2:]}"
