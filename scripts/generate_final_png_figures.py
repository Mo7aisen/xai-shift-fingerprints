#!/usr/bin/env python3
"""
Wrapper script to generate the final PNG figures used across the manuscript.

This delegates to `FigureGenerator` in `generate_final_png_figures_fixed.py`
to keep a single source of truth for figure styling and data handling.
"""

from generate_final_png_figures_fixed import FigureGenerator


def main():
    generator = FigureGenerator()
    generator.generate_all_figures()


if __name__ == '__main__':
    main()
