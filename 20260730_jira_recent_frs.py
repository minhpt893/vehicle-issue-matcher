from pathlib import Path
import re

import pandas as pd
import requests


BASE_URL = "https://tms.vinfast.vn/rest/api/2/search"
JQL = "filter=52240"
FRS_BY_VIN_FILE = Path("20260730_recent_frs_by_vin.txt")
OUTPUT_FILE = Path("20260730_issues.csv")
COOKIE_STRING = "atlassian.xsrf.token=B62F-I7LQ-1TE0-3WMR_ffe1ef1c53b2a5d9d5da229558e5c360b850ca7a_lin; _cfuvid=MvJPnjq.BOhjuA53WnmYAfXMeD.Am1EAFeXEZ8mHoYI-1783907690496-0.0.1.1-604800000; BIGipServerpool_VF-Issue-Mgmt_TCP-8080=658669578.36895.0000; JSESSIONID=448AA5A0700EC79FAD95C3BB30F8E541; TS01105ee5=012cbf5ff6e67790ced0a30ede332cd66738b238afaf952dcf87523fbd90d43f219f23d517269bc95e14788815ea6998d3c79ee20e2acd5807306c088771a948de23561b419327a21869950589e01ae8253f76a1b93bd5853de4bf517faa7d46625e748d68d22960eaed0b6f3cef07d6fbb3a15f225b139d7b1fa5bd1bc2f152855b313f05095112c77150d74099f7aafc3ea9a26bfcbaab1dd049c5cb34c7ca483aa3e19658c2c6bda474eb7e46d5a52f0cd9c1fb8c412fe748863e7d5b691a70ca799f044f9777387e2a78f1715abc339ab80375d4ca5a04c382c6bcdbc4345123385993de5535538a8ba1903797acd7f323065de3af3fa0f81b40acc3457e11427e23c84f7a1340c32b185de11948a0c17e42d8d715e9328396db26557d875f620e1d2fb8c4b687e128270dea7bd69383dffb9d60ef25face2f150164f2a68253338e636d940b26bc1916fdcc24bb4c91c2407bc9295c4b167b55338616532e6ed456e7bbd1e184d35fe4abbd9704a80c8ee5625819c323fd98b52fdf71196c4097627b34bda0cbd3bbbdf17f4d59ffa9dbb8f917d212eca386a6442d720016c375e812ad98804669270faabfab25ed08e8ab6de5d212ce2aa542ce3cb0aa511065f8db5a4010a4911026a6480c5b87770b2c39e1f26902b83194b8ab9a7e2b0de31d87b21031c985eac7c4d2964a08e95896812721d158925e96360b76e3f18658375796e9ef8301cb219d615048637817ae68e26fa26f38ddbb220d4c5e2f9bf32e7d64e03d159c34d208532cd027c5fdbda3d0e93c5e1e6c371c3a4d34377361e40303c5e84a3d5ac1cf0cf28c7da0c2a736f1862b4f799b99e86701dccfc5eea5a58f0bdc343c2cf9f70d10d4bf1e54f93d"


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


def scalar_values(value):
    """Yield scalar Jira field values whether the field is scalar or a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(scalar_values(item))
        return values
    if isinstance(value, dict):
        for key in ("value", "name", "key"):
            if value.get(key) is not None:
                return scalar_values(value[key])
        return []
    return [value]


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


def issue_frs_candidates(frs, tpv):
    """Return possible full FRS values represented by Jira's FRS/TPV fields."""
    frs = normalize_frs(frs)
    tpv = normalize_frs(tpv)
    candidates = {frs} if frs else set()

    if frs and tpv:
        candidates.add(normalize_frs(f"{frs}.{tpv}"))

    return candidates


def matching_pair(fields, allowed):
    """Return the matching (VIN, FRS) pair, or None when the issue is excluded."""
    vins = {
        normalize_vin(value)
        for value in scalar_values(fields.get("customfield_11731"))
        if normalize_vin(value)
    }

    frs = ""
    frs_field = fields.get("customfield_17403")
    if isinstance(frs_field, dict):
        frs = frs_field.get("fields", {}).get("summary", "")

    tpv = fields.get("customfield_13600")
    frs_candidates = issue_frs_candidates(frs, tpv)

    for vin in sorted(vins):
        matches = frs_candidates & allowed.get(vin, set())
        if matches:
            return vin, sorted(matches)[0], normalize_frs(frs), normalize_frs(tpv)

    return None


def format_found_datetime(value):
    if not value:
        return ""
    return value.replace("T", " ").replace(".000+0700", "GMT+7")


def main():
    allowed = load_allowed_frs_by_vin(FRS_BY_VIN_FILE)
    print(
        f"Loaded {sum(map(len, allowed.values()))} FRS entries "
        f"for {len(allowed)} VINs"
    )

    headers = {"Cookie": COOKIE_STRING, "Accept": "application/json"}
    session = requests.Session()
    all_rows = []
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
                "customfield_17403,customfield_13600"
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
                continue

            matched_vin, matched_frs, jira_frs, tpv = match
            all_rows.append(
                {
                    "Key": issue.get("key"),
                    "Summary": clean_summary(fields.get("summary")),
                    "Found datetime": format_found_datetime(
                        fields.get("customfield_17433")
                    ),
                    "FRS": matched_frs,
                    "TPV": tpv,
                    "Severity": (
                        fields.get("customfield_10226", {}).get("value")
                        if isinstance(fields.get("customfield_10226"), dict)
                        else None
                    ),
                    "VIN": matched_vin,
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
            )

        downloaded += len(issues)
        start_at += len(issues)
        print(f"Downloaded {downloaded} issues; retained {len(all_rows)}")

        total = data.get("total")
        if isinstance(total, int) and start_at >= total:
            break

    columns = [
        "Key", "Summary", "Found datetime", "FRS", "TPV", "Severity",
        "VIN", "Status", "Program",
    ]
    pd.DataFrame(all_rows, columns=columns).to_csv(
        OUTPUT_FILE, index=False, encoding="utf-8-sig"
    )
    print(f"Exported {len(all_rows)} matching issues to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()