Regenerate from repository root:
.venv/Scripts/python.exe regenerate_chapter7_figures.py

OpenAI companion to Figure 7.2:
.venv/Scripts/python.exe observations/statistics_v1-2/combined/plots/compare_summaries.py
This uses the same layout and styling with the recorded v1.2 OpenAI data, producing figure_7_2_openai and figure_7_2_openai_no_captions in PDF, SVG, and PNG, and updating the original OpenAI comparison PNG.

PDF and SVG are vector exports, exactly 152 mm wide, using embedded Noto Sans (SVG glyph paths). PNGs are previews. Font license is in fonts/LICENSE.
Sources: v1.5 comparison scripts select 4b, 8b, 30b-instruct, 30b-thinking-2; semantic/linguistic selects 30b-instruct and 30b-instruct-lingu-1; prompt comparison selects 30b-instruct and 30b-improved. Original image files are retained.
Statistics: every CSV mean and SD is verified against its stored observation arrays. Old comparison plots called the average within-label SD "Variance". The exported intervals retain those exact aggregate means and SDs, now explicitly defined. No square root conversion is appropriate.
Sampling: label weights, repeated observations, unequal Think runs, and incomplete linguistic records are retained. Intervals are descriptive, not independent-sample uncertainty estimates. Prompt sample identifiers and source run arrays are checked. Raw generation metadata are absent.
Diagnostic counts and denominators are reconstructed from case records and checked against archived summaries. Diagnostic subsets differ in size and do not cover all 510 evaluations per prompt.
See captions.txt, validation_values.csv, diagnostic_counts.csv, and data_audit.json (source hashes and sampling structure).
