import subprocess
import re
import sys
from pathlib import Path

PAPER_DIR = Path(r"E:\Desktop\DASFAA\paper")
PDFLATEX = r"E:\latex\miktex\bin\x64\pdflatex.exe"
BIBTEX = r"E:\latex\miktex\bin\x64\bibtex.exe"

def run_cmd(cmd, desc):
    print(f"--> Running {desc}...")
    res = subprocess.run(cmd, cwd=PAPER_DIR, capture_output=True, text=True, errors="ignore")
    if res.returncode != 0:
        print(f"Warning: {desc} exited with code {res.returncode}")
        # print tail of output
        lines = res.stdout.split("\n")
        print("\n".join(lines[-20:]))
    return res

def compile_all():
    # 1. pdflatex
    run_cmd([PDFLATEX, "-interaction=nonstopmode", "main.tex"], "pdflatex pass 1")

    # 2. bibtex
    run_cmd([BIBTEX, "main"], "bibtex")

    # 3. pdflatex
    run_cmd([PDFLATEX, "-interaction=nonstopmode", "main.tex"], "pdflatex pass 2")

    # 4. pdflatex
    run_cmd([PDFLATEX, "-interaction=nonstopmode", "main.tex"], "pdflatex pass 3")

    # Parse main.log
    log_file = PAPER_DIR / "main.log"
    if not log_file.exists():
        print("Error: main.log not found!")
        return

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

    # Find undefined references
    undef_refs = re.findall(r"Reference `(.*?)' on page \d+ undefined", log_content)
    undef_cits = re.findall(r"Citation `(.*?)' on page \d+ undefined", log_content)

    # Find overfull hboxes
    overfulls = []
    for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_content):
        pt = float(m.group(1))
        if pt > 1.0:
            overfulls.append(pt)

    # Find total pages
    pages_match = re.search(r"Output written on main\.pdf \((\d+) pages", log_content)
    page_count = int(pages_match.group(1)) if pages_match else 0

    print("\n" + "=" * 60)
    print("COMPILATION SUMMARY:")
    print(f"Total Pages: {page_count} (Limit: <= 16)")
    print(f"Undefined References: {len(undef_refs)} {undef_refs}")
    print(f"Undefined Citations:  {len(undef_cits)} {undef_cits}")
    print(f"Overfull Hboxes (> 1pt): {len(overfulls)} {overfulls}")
    print("=" * 60)

if __name__ == "__main__":
    compile_all()
