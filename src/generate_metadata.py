import argparse
import json
import re
from pathlib import Path


def find_abs_files(year_directories, project_root):
    """
    Recursively find all .abs files inside the given year directories.

    Returns a list of Path objects.
    """
    abs_files = []
    for year_dir in year_directories:
        if not year_dir.exists() or not year_dir.is_dir():
            continue
        abs_files.extend(year_dir.rglob("*.abs"))
    return abs_files


def _extract_folded_field(lines, field_names):
    """
    Extract a header-like field that may be folded across multiple lines.

    Folding heuristic: subsequent lines that start with whitespace are treated as
    continuations and concatenated with spaces.
    """
    field_names_set = set(field_names)
    for index, line in enumerate(lines):
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        name_stripped = name.strip()
        if name_stripped in field_names_set:
            collected = [value.strip()]
            j = index + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.startswith(" ") or next_line.startswith("\t"):
                    collected.append(next_line.strip())
                    j += 1
                    continue
                break
            return " ".join(collected).strip()
    return ""


def _extract_id_from_paper_field(paper_field_value):
    """
    Parse numeric id from values like "hep-th/9912294" or "hep-ph/0001234".
    Returns empty string if not found.
    """
    if not paper_field_value:
        return ""
    match = re.search(r"[^\s/]+/([0-9]+)", paper_field_value)
    return match.group(1) if match else ""


def _extract_abstract(lines):
    """
    Extract abstract as the block of text between the last two lines that are
    a single backslash ("\\"). If only one such line exists, use content after it.
    Falls back to empty string if nothing sensible is found.
    """
    # Treat any line that consists of one or more backslashes (optionally with spaces)
    # as a delimiter. This is more robust across archival variants.
    backslash_indices = []
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped and set(stripped) == {"\\"}:
            backslash_indices.append(i)
    if len(backslash_indices) >= 2:
        start = backslash_indices[-2] + 1
        end = backslash_indices[-1]
    elif len(backslash_indices) == 1:
        start = backslash_indices[0] + 1
        end = len(lines)
    else:
        return ""

    abstract_lines = [lines[k].rstrip("\n\r") for k in range(start, max(start, end))]
    # Trim leading/trailing empty lines
    while abstract_lines and not abstract_lines[0].strip():
        abstract_lines.pop(0)
    while abstract_lines and not abstract_lines[-1].strip():
        abstract_lines.pop()
    return "\n".join(abstract_lines)


def parse_abs_file(file_path):
    """
    Parse a single .abs file to extract id (numeric part only), title, authors,
    and abstract.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, FileNotFoundError):
        return None

    lines = text.splitlines()

    paper_field = _extract_folded_field(lines, ["Paper"])
    title = _extract_folded_field(lines, ["Title"])
    # Support both "Authors" and "Author"
    authors = _extract_folded_field(lines, ["Authors", "Author"])

    paper_id = _extract_id_from_paper_field(paper_field)
    abstract = _extract_abstract(lines)

    return {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate JSON with id, title, authors, abstract from .abs files.")
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Years to include (default: 1992..2003)",
    )
    parser.add_argument(
        "--out",
        default="abs_metadata.json",
        help="Output JSON file path (default: abs_metadata.json)",
    )

    args = parser.parse_args(argv)

    project_root = Path(args.root).resolve()

    if args.years and len(args.years) > 0:
        years = args.years
    else:
        years = list(range(1992, 2004))

    year_directories = [project_root / str(y) for y in years]

    abs_files = find_abs_files(year_directories, project_root)

    # Build records with desired fields only
    records = []
    for file_path in abs_files:
        record = parse_abs_file(file_path)
        if record is None:
            continue
        # Skip entries that lack an id (cannot determine from file)
        if not record.get("id"):
            continue
        records.append(record)

    output_path = (project_root / args.out).resolve()
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()


