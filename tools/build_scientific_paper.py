#!/usr/bin/env python3
"""Compile the scientific article and reject unresolved or overflowing TeX output."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def build(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if not (source / 'main.tex').is_file():
        raise ValueError('Scientific main.tex is missing')
    compiler = shutil.which('latexmk')
    if compiler is None:
        raise ValueError('Install latexmk and pdflatex to compile the paper')
    # Compile a separate copy: never commit TeX scratch or read an old local PDF.
    with tempfile.TemporaryDirectory(prefix='tlab-paper-') as temporary:
        stage = Path(temporary)
        inputs = {}
        for path in sorted(source.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlinked article sources are not supported')
            if path.is_file() and path.suffix in {'.tex', '.bib', '.pdf', '.png', '.jpg', '.jpeg', '.json'} and path.name != 'main.pdf':
                relative = path.relative_to(source)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                inputs[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = subprocess.run([compiler, '-pdf', '-interaction=nonstopmode',
                                 '-halt-on-error', '-file-line-error', 'main.tex'],
                                cwd=stage, text=True, capture_output=True, timeout=120)
        log = (stage / 'main.log').read_text(errors='replace') if (stage / 'main.log').exists() else ''
        problems = re.findall(r'^.*(?:Overfull \\[hv]box|undefined references|Citation .* undefined|LaTeX Error|^!).*$', log, re.M)
        if result.returncode or problems:
            raise ValueError('Paper compilation failed: ' + '\n'.join(problems) + '\n' + result.stdout[-3000:] + result.stderr[-1000:])
        pdf = stage / 'main.pdf'
        if not pdf.is_file() or not pdf.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Compiler did not produce a PDF')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, output)
        provenance = {'source_files': inputs, 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                      'compiler': 'latexmk -pdf', 'checks': {'compiled': True, 'no_overflow': True, 'no_undefined_references': True}}
        (output.parent / 'PDF_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('paper/submission'))
    parser.add_argument('--output', type=Path, default=Path('reports/submission/main.pdf'))
    args = parser.parse_args()
    print(build(args.source, args.output))
