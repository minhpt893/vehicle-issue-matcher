from pathlib import Path
import re

import pandas as pd
import requests


BASE_URL = "https://tms.vinfast.vn/rest/api/2/search"
JQL = "filter=52240"
FRS_BY_VIN_FILE = Path("20260730_recent_frs_by_vin.txt")
OUTPUT_FILE = Path("20260730_issues.csv")
NOT_RETAINED_OUTPUT_FILE = Path("20260730_issues_not_retained.csv")
COOKIE_STRING = ""
SCRIPT_VERSION = "2026-07-22-not-retained-output"


def normalize_vin(value):
    """Normalize a VIN for exact, case-insensitive comparison."""
    return str(value or "").strip().upper()


def normalize_frs(value):
    """Normalize insignificant formatting while retaining the FRS structure."""
    value = str(value or "").strip().upper()
    value = re.sub(r"^FRS\s*", "", value)
    value = re.sub(r"\s+", "", value)
    return value.strip(".")


def load_allowed_frs_by_vin(path):
    """Parse VIN sections and their numbered FRS entries from a text file."""
    if not path.is_file():
        raise FileNotFoundError(f"FRS-by-VIN file not found: {path.resolve()}")

    allowed = {}
    current_vin = None

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            vin_match = re.fullmatch(r"VIN\s*:\s*(.+)", line, flags=re.IGNORECASE)
            if vin_match:
                current_vin = normalize_vin(vin_match.group(1))
                if not current_vin:
                    raise ValueError(f"Empty VIN on line {line_number}")
                allowed.setdefault(current_vin, set())
                continue

            frs_match = re.fullmatch(r"\d+\.\s*(.+)", line)
            if frs_match:
                if current_vin is None:
                    raise ValueError(
                        f"FRS entry appears before the first VIN on line {line_number}"
                    )
                frs = normalize_frs(frs_match.group(1))
                if not frs:
                    raise ValueError(f"Empty FRS on line {line_number}")
                allowed[current_vin].add(frs)
                continue

            raise ValueError(
                f"Unrecognized line {line_number} in {path.name}: {raw_line.rstrip()!r}"
            )

    if not allowed:
        raise ValueError(f"No VIN/FRS entries found in {path.name}")

    empty_vins = [vin for vin, frs_values in allowed.items() if not frs_values]
    if empty_vins:
        raise ValueError(f"VIN(s) without an FRS entry: {', '.join(empty_vins)}")

    return allowed


def clean_summary(summary):
    """Keep Vietnamese text, ignoring slashes inside square brackets."""
    if not summary:
        return ""

    bracket_depth = 0
    for index, character in enumerate(summary):
        if character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        elif character == "/" and bracket_depth == 0:
            return summary[:index].strip()

    return summary.strip()


def matching_pair(fields, allowed):
    """Return the matching (VIN, FRS) pair, or None when the issue is excluded."""
    vin = normalize_vin(fields.get("customfield_11731"))
    if not vin:
        return None

    frs = ""
    frs_field = fields.get("customfield_17403")
    if isinstance(frs_field, dict):
        frs = frs_field.get("fields", {}).get("summary", "")

    frs = normalize_frs(frs)
    if frs and frs in allowed.get(vin, set()):
        return vin, frs

    return None


def get_issue_frs(fields):
    """Extract and normalize the issue's scalar Jira FRS value."""
    frs_field = fields.get("customfield_17403")
    if not isinstance(frs_field, dict):
        return ""
    return normalize_frs(frs_field.get("fields", {}).get("summary", ""))


def format_found_datetime(value):
    if not value:
        return ""
    return value.replace("T", " ").replace(".000+0700", "GMT+7")


def issue_to_row(issue, fields, vin, frs):
    """Convert a Jira issue into one output CSV row."""
    return {
        "Key": issue.get("key"),
        "Summary": clean_summary(fields.get("summary")),
        "Found datetime": format_found_datetime(fields.get("customfield_17433")),
        "FRS": frs,
        "Severity": (
            fields.get("customfield_10226", {}).get("value")
            if isinstance(fields.get("customfield_10226"), dict)
            else None
        ),
        "VIN": vin,
        "Status": (
            fields.get("status", {}).get("name")
            if isinstance(fields.get("status"), dict)
            else None
        ),
        "Program": (
            fields.get("customfield_10800", {}).get("value")
            if isinstance(fields.get("customfield_10800"), dict)
            else None
        ),
    }


def main():
    print(f"Running Jira-to-CSV downloader: {SCRIPT_VERSION}")
    allowed = load_allowed_frs_by_vin(FRS_BY_VIN_FILE)
    print(
        f"Loaded {sum(map(len, allowed.values()))} FRS entries "
        f"for {len(allowed)} VINs"
    )

    headers = {"Cookie": COOKIE_STRING, "Accept": "application/json"}
    session = requests.Session()
    all_rows = []
    not_retained_rows = []
    start_at = 0
    downloaded = 0

    while True:
        params = {
            "jql": JQL,
            "startAt": start_at,
            "maxResults": 1000,
            "fieldsByKeys": "false",
            "fields": (
                "summary,customfield_17433,customfield_10226,"
                "customfield_11731,status,customfield_10800,"
                "customfield_17403"
            ),
        }

        response = session.get(BASE_URL, params=params, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        issues = data.get("issues", [])

        if not issues:
            break

        for issue in issues:
            fields = issue.get("fields", {})
            match = matching_pair(fields, allowed)
            if match is None:
                not_retained_rows.append(
                    issue_to_row(
                        issue,
                        fields,
                        normalize_vin(fields.get("customfield_11731")),
                        get_issue_frs(fields),
                    )
                )
                continue

            matched_vin, matched_frs = match
            all_rows.append(
                issue_to_row(issue, fields, matched_vin, matched_frs)
            )

        downloaded += len(issues)
        start_at += len(issues)
        print(
            f"Downloaded {downloaded} issues; retained {len(all_rows)}; "
            f"not retained {len(not_retained_rows)}"
        )

        total = data.get("total")
        if isinstance(total, int) and start_at >= total:
            break

    columns = [
        "Key", "Summary", "Found datetime", "FRS", "Severity",
        "VIN", "Status", "Program",
    ]
    pd.DataFrame(all_rows, columns=columns).to_csv(
        OUTPUT_FILE, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(not_retained_rows, columns=columns).to_csv(
        NOT_RETAINED_OUTPUT_FILE, index=False, encoding="utf-8-sig"
    )
    print(f"Exported {len(all_rows)} matching issues to {OUTPUT_FILE}")
    print(
        f"Exported {len(not_retained_rows)} non-matching issues "
        f"to {NOT_RETAINED_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()