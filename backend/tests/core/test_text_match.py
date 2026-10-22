"""Tests for shared email text-matching / noise-stripping helpers."""

import re

from core.text_match import meaningful_tokens, strip_email_noise, strip_synthesis_noise


def test_strip_email_noise_drops_external_banner_keeps_body():
    body = (
        "EXTERNAL EMAIL: Do not click any links or open any attachments "
        "unless you trust the sender and know the content is safe.\n"
        "Please send the WB CASA numbers by Friday."
    )
    cleaned = strip_email_noise(body)
    assert "EXTERNAL EMAIL" not in cleaned
    assert "WB CASA numbers" in cleaned


def test_strip_email_noise_drops_quoted_original_message():
    body = (
        "Here is the updated deposit mix.\n"
        "'-----Original Message-----\n"
        "From: someone@rblbank.com\n"
        "Sep-26 CA/SA targets - WB view required\n"
    )
    cleaned = strip_email_noise(body)
    assert "updated deposit mix" in cleaned
    assert "CA/SA targets" not in cleaned


def test_strip_synthesis_noise_keeps_thread_drops_from_to_cc():
    body = (
        "EXTERNAL EMAIL: Do not click any links.\n"
        "From: Sameer Kapoor <sameer.kapoor@rbl.bank.in>\n"
        "To: digitaloffice@rblbank.com\n"
        "Cc: Chourishi, Tanishq <tchourishi@alvarezandmarsal.com>\n\n"
        "Risk-review time is 4.2 days; rework and queueing are 9+ days combined.\n"
        "-----Original Message-----\n"
        "From: Amit Sethi <amit.sethi@rbl.bank.in>\n"
        "Average elapsed time on an exception case is 13.4 days.\n"
    )
    cleaned = strip_synthesis_noise(body)
    assert "4.2 days" in cleaned
    assert "13.4 days" in cleaned
    assert "EXTERNAL EMAIL" not in cleaned
    assert "Chourishi" not in cleaned
    assert not re.search(r"^From:", cleaned, re.MULTILINE)


def test_meaningful_tokens_treat_ca_sa_slash_as_casa():
    tokens = meaningful_tokens("Sep-26 CA/SA targets - WB view required")
    assert "casa" in tokens
    assert "wholesale" in tokens
    assert "banking" in tokens
    assert "targets" in tokens
