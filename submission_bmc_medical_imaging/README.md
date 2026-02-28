# BMC Medical Imaging Overleaf Package

This folder is the BMC Medical Imaging submission package prepared from the canonical project sources on `2026-02-28`.

## Purpose

- Target journal: `BMC Medical Imaging`
- Primary contribution: Phase 1 Extended
- Hardshift branch: retained as an honest negative-result extension with reviewer-traceable governance files

## Main files

- `main.tex`: single-file Springer Nature manuscript source (`sn-jnl` template)
- `sn-jnl.cls`: official Springer Nature class file
- `references.bib`: bibliography database
- `figures/*.pdf`: manuscript figures
- `supplementary/*`: reviewer-traceable supplementary files, including freeze and governance records
- `cover_letter.tex`: BMC-oriented cover letter draft

## Overleaf compile

Use `pdfLaTeX` + `BibTeX`.

Typical local build:

```bash
cd submission_bmc_medical_imaging
latexmk -pdf -interaction=nonstopmode main.tex
```

If `latexmk` is unavailable:

```bash
cd submission_bmc_medical_imaging
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes

- The manuscript is intentionally kept as a single `.tex` file to match Springer Nature template guidance.
- The code availability statement includes the public GitHub repository URL directly in the paper.
- The hardshift extension is documented as a negative result and should not be framed as a positive deployment claim during submission.
