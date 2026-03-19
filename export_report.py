"""Export a markdown report to a standalone HTML file."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Markdown report to HTML.")
    parser.add_argument(
        "--input-md",
        type=Path,
        required=True,
        help="Input markdown file.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to the markdown path with an .html suffix.",
    )
    return parser.parse_args()


def _inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    blocks: list[str] = []
    in_list = False
    in_code = False
    in_table = False
    table_lines: list[str] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            blocks.append("</ul>")
            in_list = False

    def close_table() -> None:
        nonlocal in_table, table_lines
        if not in_table:
            return
        header_cells = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
        rows = table_lines[2:] if len(table_lines) > 2 else []
        blocks.append("<table>")
        blocks.append("<thead><tr>" + "".join(f"<th>{_inline_markup(cell)}</th>" for cell in header_cells) + "</tr></thead>")
        blocks.append("<tbody>")
        for row in rows:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            blocks.append("<tr>" + "".join(f"<td>{_inline_markup(cell)}</td>" for cell in cells) + "</tr>")
        blocks.append("</tbody></table>")
        in_table = False
        table_lines = []

    for line in lines:
        if line.startswith("```"):
            close_list()
            close_table()
            if in_code:
                blocks.append("</code></pre>")
                in_code = False
            else:
                blocks.append("<pre><code>")
                in_code = True
            continue

        if in_code:
            blocks.append(html.escape(line))
            continue

        if line.startswith("|") and line.endswith("|"):
            close_list()
            in_table = True
            table_lines.append(line)
            continue
        close_table()

        if not line.strip():
            close_list()
            blocks.append("")
            continue

        if line.startswith("### "):
            close_list()
            blocks.append(f"<h3>{_inline_markup(line[4:])}</h3>")
        elif line.startswith("## "):
            close_list()
            blocks.append(f"<h2>{_inline_markup(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list()
            blocks.append(f"<h1>{_inline_markup(line[2:])}</h1>")
        elif line.startswith("- "):
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{_inline_markup(line[2:])}</li>")
        else:
            close_list()
            blocks.append(f"<p>{_inline_markup(line)}</p>")

    close_list()
    close_table()
    if in_code:
        blocks.append("</code></pre>")

    body = "\n".join(block for block in blocks if block is not None)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Analysis Report</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --paper: #fffdf8;
      --ink: #1f2933;
      --muted: #52606d;
      --accent: #7c3aed;
      --grid: #d9d2c3;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background: radial-gradient(circle at top, #fffaf0, var(--bg));
      color: var(--ink);
    }}
    main {{
      max-width: 900px;
      margin: 40px auto;
      padding: 40px;
      background: var(--paper);
      box-shadow: 0 18px 60px rgba(31, 41, 51, 0.12);
      border: 1px solid rgba(124, 58, 237, 0.08);
    }}
    h1, h2, h3 {{ line-height: 1.15; }}
    h1 {{ font-size: 2.4rem; }}
    h2 {{
      margin-top: 2rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--grid);
    }}
    p, li {{ font-size: 1.02rem; line-height: 1.65; }}
    code {{
      background: #f1ebff;
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
      font-size: 0.94em;
    }}
    pre {{
      overflow-x: auto;
      background: #191724;
      color: #f8fafc;
      padding: 1rem;
      border-radius: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.96rem;
    }}
    th, td {{
      border: 1px solid var(--grid);
      padding: 0.55rem 0.7rem;
      text-align: left;
    }}
    th {{
      background: #f8f4ee;
    }}
  </style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_md = args.input_md.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve() if args.output_html else input_md.with_suffix(".html")
    markdown_text = input_md.read_text(encoding="utf-8")
    output_html.write_text(markdown_to_html(markdown_text), encoding="utf-8")
    print(f"Wrote HTML report to {output_html}")


if __name__ == "__main__":
    main()
