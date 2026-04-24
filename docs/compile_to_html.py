#!/usr/bin/env python3
"""build_docs.py — Convert markdown workflow docs to styled HTML.

Usage:
    python3 docs/build_docs.py                  # convert all .md files in docs/
    python3 docs/build_docs.py workflow_fb.md    # convert one file

Requires: pip install markdown
"""

import base64
import glob
import mimetypes
import os
import re
import sys

import markdown

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))

CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 860px; margin: 0 auto; padding: 24px 32px; color: #222;
       line-height: 1.6; }
h1 { border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }
h2 { margin-top: 36px; color: #2c3e50; }
h3 { color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 28px 0; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.92em; }
pre { background: #1e1e2e; color: #cdd6f4; padding: 14px 18px; border-radius: 6px;
      overflow-x: auto; font-size: 0.88em; line-height: 1.5; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #e0c060; background: #fffde7; margin: 16px 0;
             padding: 10px 16px; border-radius: 0 4px 4px 0; }
blockquote p { margin: 4px 0; }
ul, ol { padding-left: 24px; }
li { margin: 4px 0; }
a { color: #2980b9; }
table { border-collapse: collapse; margin: 12px 0; font-size: 0.92em; }
th, td { border: 1px solid #ddd; padding: 6px 12px; }
th { background: #f5f5f5; text-align: left; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }
h1 .anchor, h2 .anchor, h3 .anchor, h4 .anchor {
    text-decoration: none; color: #ccc; margin-left: 6px; font-size: 0.8em;
}
h1:hover .anchor, h2:hover .anchor, h3:hover .anchor, h4:hover .anchor {
    color: #2980b9;
}
"""

_ANCHOR_SCRIPT = """\
<script>
document.querySelectorAll('h1[id],h2[id],h3[id],h4[id]').forEach(h => {
    const a = document.createElement('a');
    a.className = 'anchor'; a.href = '#' + h.id; a.textContent = '#';
    h.appendChild(a);
});
</script>
"""

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head><body>
{body}
{anchor_script}
</body></html>
"""


def md_to_html(md_path: str) -> str:
    with open(md_path) as f:
        md_text = f.read()

    # Extract title from first H1
    title_match = re.match(r'^#\s+(.+)', md_text)
    title = title_match.group(1) if title_match else os.path.basename(md_path)

    # Strip {#id} anchors from markdown (used for explicit section IDs)
    # and convert to attr_list format that the markdown extension understands
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "attr_list"],
    )

    return HTML_TEMPLATE.format(title=title, css=CSS, body=body,
                                anchor_script=_ANCHOR_SCRIPT)


def _inline_images(html: str, base_dir: str) -> str:
    """Replace local image src with inline base64 data URIs."""
    # Try resolving relative to md file dir, then repo root
    repo_root = os.path.abspath(os.path.join(DOCS_DIR, '..', '..'))

    def _replace(m):
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        # Try multiple base directories
        for bd in [base_dir, repo_root]:
            img_path = os.path.normpath(os.path.join(bd, src))
            if os.path.exists(img_path):
                mime = mimetypes.guess_type(img_path)[0] or "image/png"
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return f'src="data:{mime};base64,{b64}"'
        return m.group(0)
    return re.sub(r'src="([^"]+)"', _replace, html)


def convert_file(md_path: str):
    html_path = md_path.rsplit(".", 1)[0] + ".html"
    html = md_to_html(md_path)
    html = _inline_images(html, os.path.dirname(md_path))
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  {os.path.basename(md_path)} -> {os.path.basename(html_path)}")


def main():
    if len(sys.argv) > 1:
        files = [os.path.join(DOCS_DIR, f) if not os.path.isabs(f) else f
                 for f in sys.argv[1:]]
    else:
        files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))

    if not files:
        print("No .md files found in docs/")
        return

    print(f"Converting {len(files)} file(s):")
    for md_path in files:
        if not os.path.exists(md_path):
            print(f"  WARNING: {md_path} not found, skipping")
            continue
        convert_file(md_path)
    print("Done.")


if __name__ == "__main__":
    main()
