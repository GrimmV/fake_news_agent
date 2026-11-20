"""Render the OpenAI model comparison in the style of Qwen3 Figure 7.2.

Run from the repository root:
.venv/Scripts/python.exe observations/statistics_v1-2/combined/plots/compare_summaries.py

Means and within-label sample SDs are verified against the stored observations.
"""
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from regenerate_chapter7_figures import (
    BLUE,
    ORANGE,
    OUT,
    comparison,
    read_config,
    style,
)


DATA = ROOT / 'observations/statistics_v1-2'
CONFIGURATIONS = ['gpt-4-1-nano', 'gpt-4-1-mini', 'gpt-5-mini']
SETTING_LABELS = ['GPT-4.1\nNano', 'GPT-4.1\nMini', 'GPT-5\nMini']
OUTPUT_FIGURE = Path(__file__).with_name(
    'compare_summaries-gpt-4-1-nano-gpt-4-1-mini-gpt-5-mini.png'
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    configs = [read_config(name, data_dir=DATA) for name in CONFIGURATIONS]
    for include_captions in [True, False]:
        comparison(
            '2_openai', configs, SETTING_LABELS,
            [BLUE, BLUE, ORANGE], ['o', 'o', 'D'],
            'OpenAI model configurations',
            [(BLUE, 'o', 'Standard'), (ORANGE, 'D', 'Reasoning enabled')],
            include_captions=include_captions,
        )
    # Keep the existing image path usable by documents that already reference it.
    shutil.copyfile(OUT / 'figure_7_2_openai.png', OUTPUT_FIGURE)
    print(f'Saved OpenAI PDF, SVG, and PNG variants to: {OUT}')
    print(f'Updated original comparison image: {OUTPUT_FIGURE}')


if __name__ == '__main__':
    main()
