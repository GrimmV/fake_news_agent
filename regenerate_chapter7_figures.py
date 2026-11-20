"""Reproduce Figures 7.2–7.5 from the recorded v1.5 results.

Run: .venv/Scripts/python.exe regenerate_chapter7_figures.py
No generation, imputation, reweighting, or overwriting of original results.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/chapter7'
DATA = ROOT / 'observations/statistics_v1-5'
BLUE, ORANGE, TEAL, GRAY = '#0072B2', '#E69F00', '#009E73', '#666666'
METRICS = [
    ('xai_description_truthfulness', 'Source-to-description\ntruthfulness'),
    ('technical_xai_clarity', 'Technical-description clarity'),
    ('layman_xai_truthfulness', 'Technical-to-layperson\ntruthfulness'),
    ('label_correlation', 'Dataset-label alignment'),
    ('technical_assessment_clarity', 'Assessment-rationale clarity'),
    ('short_assessment_truthfulness', 'Technical-to-short-rationale\ntruthfulness'),
]
CATEGORIES = {
    'feature_interpretation_bias': 'Feature-interpretation bias',
    'meta_performance_overweighting': 'Meta-performance overweighting',
    'explanation_framing': 'Explanation framing',
    'label_trust_mismatch': 'Label–assessment mismatch',
    'overgeneralization_from_dataset_statistics': 'Global-to-local overgeneralization',
    'content_model_confusion': 'Content–model confusion',
    'performance_baseline': 'Performance baseline',
}
AUDIT = {'dispersion': 'Arithmetic mean of six label-specific sample SDs (ddof=1); not variance, pooled SD, SE, or CI.', 'sources': {}, 'configurations': {}}
NUMBERS = []


def source(path):
    raw = path.read_bytes()
    AUDIT['sources'][path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8-sig')


def literal(path):
    tree = ast.parse(source(path))
    return ast.literal_eval(tree.body[0].value)


def read_config(name, data_dir=DATA):
    summary = list(csv.DictReader(source(data_dir / 'combined/plots' / f'set-{name}_summary.csv').splitlines()))
    result = {}
    shapes = {}
    for metric, _ in METRICS:
        values = literal(data_dir / 'combined' / f'set-{name}' / f'{metric}.py')
        shapes[metric] = {label: [len(row) for row in rows] for label, rows in values.items()}
        means, sds = [], []
        for label, rows in values.items():
            a = np.array([v for row in rows for v in row], dtype=float)
            assert len(a) > 1 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
            saved = next(r for r in summary if r['metric'] == metric and r['label'] == label)
            mean, sd = a.mean(), a.std(ddof=1)
            np.testing.assert_allclose([mean, sd], [float(saved['mean']), float(saved['std'])], rtol=1e-12, atol=1e-14)
            means.append(mean)
            sds.append(sd)
        # Preserve the existing comparison scripts' equal weighting of labels.
        mean, sd = float(np.mean(means)), float(np.mean(sds))
        result[metric] = (mean, sd)
        NUMBERS.append(dict(configuration=name, metric=metric, mean=mean,
                            mean_within_label_sample_sd=sd, lower=mean-sd, upper=mean+sd,
                            n=sum(sum(s) for s in shapes[metric].values())))
    AUDIT['configurations'][name] = {'recorded_shapes': shapes}
    return result


def style():
    for weight in ['Regular', 'Bold']:
        font = OUT / 'fonts' / f'NotoSans-{weight}.ttf'
        if not font.exists():
            raise FileNotFoundError(f'Required Noto Sans font: {font}')
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Noto Sans', 'font.size': 9,
                         'text.color': '#222222', 'axes.labelcolor': '#222222',
                         'axes.edgecolor': GRAY, 'axes.linewidth': .65,
                         'xtick.color': GRAY, 'ytick.color': GRAY,
                         'xtick.labelcolor': '#222222', 'ytick.labelcolor': '#222222',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'pdf.fonttype': 42, 'svg.fonttype': 'path'})


def save(fig, name, no_captions=True):
    # Do not use bbox_inches='tight': the exported page must remain exactly 152 mm wide.
    suffix = '_no_captions' if no_captions else ''
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}{suffix}.{ext}', dpi=220)
    plt.close(fig)


def comparison(number, configs, labels, colors, markers, title, legend=None, *, include_captions=False):
    fig, axes = plt.subplots(3, 2, figsize=(152/25.4, 190/25.4))
    fig.subplots_adjust(left=.095, right=.975, bottom=.085 if include_captions else .065,
                        top=.79 if include_captions else .825, wspace=.30, hspace=.78)
    if include_captions:
        fig.suptitle(title, y=.975, fontsize=10, weight='bold')
        fig.text(.5, .025, 'Mean ± mean within-label SD', ha='center', fontsize=9)
    headings = ['Source-near and intermediate\ntransformations', 'High-level assessment\nand rationale']
    for col, heading in enumerate(headings):
        box = axes[0, col].get_position()
        fig.text((box.x0+box.x1)/2, .865 if include_captions else .90,
                 heading, ha='center', fontsize=9.5, weight='bold')
    if legend:
        handles = [Line2D([], [], color=c, marker=m, linestyle='none', label=l, markersize=5)
                   for c, m, l in legend]
        fig.legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(.5, .945 if include_captions else .995),
                   ncol=2, frameon=False, fontsize=9)
    for i, (metric, heading) in enumerate(METRICS):
        ax = axes[i % 3, i // 3]
        for x, (config, color, marker) in enumerate(zip(configs, colors, markers)):
            mean, sd = config[metric]
            ax.errorbar(x, mean, yerr=sd, fmt=marker, color=color, markersize=5,
                        capsize=3, capthick=1, elinewidth=1.15, linestyle='none')
        ax.set_title(heading, fontsize=9, loc='left', pad=8)
        ax.set_xticks(range(len(labels)), labels)
        ax.tick_params(axis='both', labelsize=9, length=3)
        ax.set_xlim(-.5, len(labels)-.5)
        # SD intervals can extend beyond the score bounds. Show them in full,
        # with the same axis limits in all panels and all three figures.
        ax.set_ylim(-.05, 1.3)
        ax.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    fig.text(.018, .46, 'Score (0–1)', rotation=90, va='center', fontsize=9)
    save(fig, f'figure_7_{number}', no_captions=not include_captions)


def diagnoses():
    datasets = []
    for name in ['initial', 'improved']:
        raw = json.loads(source(ROOT / f'experiment_results-{name}_v1.5.json'))
        stored = json.loads(source(ROOT / f'cause_statistics-{name}_v1.5.json'))
        cases = [c for rows in raw.values() for c in rows]
        flagged = [c for c in cases if c.get('should_investigate')]
        assert all(c.get('analysis') and c.get('case_type') for c in flagged)
        primary, secondary = Counter(), Counter()
        for case in flagged:
            analysis = case['analysis']['analysis']
            assert analysis.get('primary_cause')
            assert len(analysis['secondary_causes']) == len(set(analysis['secondary_causes']))
            primary[analysis['primary_cause']] += 1
            secondary.update(analysis['secondary_causes'])
        assert sum(primary.values()) == len(flagged) == sum(s['total_cases'] for s in stored.values())
        for kind, counts in [('primary', primary), ('secondary', secondary)]:
            expected = Counter()
            for entry in stored.values():
                expected.update(entry[f'{kind}_causes'])
            assert counts == expected, (name, kind)
        datasets.append(dict(name=name, total=len(flagged), evaluated=len(cases),
                             primary=primary, secondary=secondary))
    unknown = set().union(*(set(d[k]) for d in datasets for k in ['primary', 'secondary'])) - CATEGORIES.keys()
    assert unknown <= {'other'}, unknown
    categories = dict(CATEGORIES)
    if 'other' in unknown:
        categories['other'] = 'Other'
    fig, axes = plt.subplots(2, 1, figsize=(152/25.4, 190/25.4), sharex=True)
    fig.subplots_adjust(left=.47, right=.95, top=.88, bottom=.08, hspace=.32)
    max_count = max(v for d in datasets for k in ['primary', 'secondary'] for v in d[k].values())
    rows = []
    for ax, d, title in zip(axes, datasets, ['(a) Original prompt', '(b) Revised prompt']):
        y = np.arange(len(categories))
        for offset, kind, color in [(-.18, 'primary', BLUE), (.18, 'secondary', TEAL)]:
            counts = [d[kind][k] for k in categories]
            bars = ax.barh(y+offset, counts, height=.31, color=color, label=f'{kind.title()} causes')
            ax.bar_label(bars, padding=3, fontsize=9)
            for key, count in zip(categories, counts):
                rows.append(dict(prompt=title[4:], category=categories[key], diagnosis=kind,
                                 count=count, divergent_denominator=d['total'],
                                 percent_of_divergent=100*count/d['total']))
        ax.set_yticks(y, categories.values())
        ax.invert_yaxis()
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_title(f"{title}\n{d['total']} divergent assessments", fontsize=10, loc='left', pad=12)
        ax.set_xlim(0, np.ceil((max_count+6)/10)*10)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#eeeeee', linewidth=.5)
    axes[1].set_xlabel('Diagnostic count')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .995), ncol=2, frameon=False)
    save(fig, 'figure_7_4')
    write_csv('diagnostic_counts.csv', rows)
    AUDIT['diagnostics'] = datasets
    return datasets


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_prompt_sample():
    labels = list(literal(DATA / 'combined/set-30b-instruct/label_correlation.py'))
    sample = {}
    for label in labels:
        sequences = []
        for old, new in zip(range(1, 6), range(36, 41)):
            for group, run in [('30b-instruct', old), ('30b-improved', new)]:
                for metric, _ in METRICS:
                    combined = literal(DATA / f'combined/set-{group}/{metric}.py')[label]
                    original = literal(DATA / f'set_{run}/{metric}.py')[label]
                    assert combined[(old-1)] == original
                records = json.loads(source(ROOT / f'observations/experiments_raw_v1.5/{label}/exp-{run}.json'))
                sequences.append([r['input']['dp_id'] for r in records])
        assert all(seq == sequences[0] for seq in sequences)
        sample[label] = sequences[0]
    AUDIT['prompt_sample'] = {'matching_dp_ids_by_label': sample,
                             'original_runs': list(range(1, 6)), 'revised_runs': list(range(36, 41)),
                             'limitation': 'Raw exports do not contain model identity, prompt text, or temperature metadata; fixed generation settings cannot be independently verified from these exports.'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    names = ['4b', '8b', '30b-instruct', '30b-thinking-2', '30b-instruct-lingu-1', '30b-improved']
    data = {name: read_config(name) for name in names}
    verify_prompt_sample()
    comparison(2, [data[n] for n in names[:4]], ['4B', '8B', '30B\nInstruct', '30B\nThink'],
               [BLUE]*3+[ORANGE], ['o']*3+['D'], 'Qwen3 model configurations',
               [(BLUE, 'o', 'Standard'), (ORANGE, 'D', 'Reasoning enabled')])
    comparison(3, [data[n] for n in ['30b-instruct', '30b-instruct-lingu-1']],
               ['Semantic', 'Linguistic'], [BLUE, ORANGE], ['o', 's'], 'Semantic and linguistic evidence representations',
               [(BLUE, 'o', 'Semantic'), (ORANGE, 's', 'Linguistic')])
    ds = diagnoses()
    comparison(5, [data[n] for n in ['30b-instruct', '30b-improved']],
               ['Original\nprompt', 'Revised\nprompt'], [GRAY, BLUE], ['o', 's'], 'Evaluation before and after prompt refinement',
               [(GRAY, 'o', 'Original prompt'), (BLUE, 's', 'Revised prompt')])
    write_csv('validation_values.csv', NUMBERS)
    (OUT / 'data_audit.json').write_text(json.dumps(AUDIT, indent=2), encoding='utf-8')
    dispersion = ('Points show the equally weighted mean of the six dataset-label means; intervals show ± the arithmetic mean of the six within-label sample standard deviations (ddof=1), preserving the original aggregation. This dispersion is not variance, a pooled SD, or a confidence interval. Scores range from 0 to 1; axes extend from -0.05 to 1.3 to display SD intervals without truncation. ')
    columns = ('The left column contains source-near and intermediate transformations; the right column contains high-level assessment and rationale measures. ')
    captions = [
        'Figure 7.2. Comparison of Qwen3 model configurations: 4B, 8B, 30B Instruct (standard configurations), and 30B Think (reasoning enabled). '+dispersion+columns+'The recorded data comprise five runs of 17 cases per label for each standard configuration (510 evaluations each) and four such runs for Think (408 evaluations).',
        'Figure 7.3. Comparison of evidence-representation configurations using Qwen3 30B Instruct without reasoning. Semantic uses domain-oriented semantic features; Linguistic uses primarily deterministic linguistic and stylistic features. The upstream representation, classifier inputs, and associated XAI evidence change together; this comparison does not isolate semantic similarity as a causal variable. '+dispersion+columns+'Semantic contains 85 observations per label (510 total). Linguistic contains 68 per label except pof, which contains 37 (377 total); all recorded observations and equal label weights are retained.',
        f"Figure 7.4. Diagnostic categories of divergent Advisor assessments under (a) the original prompt and (b) the revised prompt. A primary cause is the main diagnosis for an assessment; multiple secondary causes may co-occur. Counts are reconstructed from the original case records and verified against the recorded diagnostic summaries. The original configuration contains {ds[0]['total']} divergent assessments among {ds[0]['evaluated']} analyzed assessments; the revised configuration contains {ds[1]['total']} among {ds[1]['evaluated']}. Category percentages in diagnostic_counts.csv use {ds[0]['total']} and {ds[1]['total']}, respectively, as denominators. Diagnostic records cover unequal subsets of the evaluation runs; absolute counts are not directly comparable rates. The figure describes category frequencies and redistribution, not elimination of all divergence types.",
        'Figure 7.5. Evaluation of a targeted revision to the assessment-generation prompt, comparing Original prompt and Revised prompt under the study’s Qwen3 30B Instruct configuration without reasoning. '+dispersion+columns+'Both conditions retain the same 17 case identifiers per label across five runs (510 evaluations per condition); no new generations or sample substitutions were made. The archived exports do not include model identity or generation-setting metadata, so constancy of those settings cannot be independently verified from these files.'
    ]
    (OUT / 'captions.txt').write_text('\n\n'.join(captions)+'\n', encoding='utf-8')
    (OUT / 'README.txt').write_text(
        'Regenerate from repository root:\n.venv/Scripts/python.exe regenerate_chapter7_figures.py\n\n'
        'OpenAI companion to Figure 7.2:\n.venv/Scripts/python.exe observations/statistics_v1-2/combined/plots/compare_summaries.py\n'
        'This uses the same layout and styling with the recorded v1.2 OpenAI data, producing figure_7_2_openai and figure_7_2_openai_no_captions in PDF, SVG, and PNG, and updating the original OpenAI comparison PNG.\n\n'
        'PDF and SVG are vector exports, exactly 152 mm wide, using embedded Noto Sans (SVG glyph paths). PNGs are previews. Font license is in fonts/LICENSE.\n'
        'Sources: v1.5 comparison scripts select 4b, 8b, 30b-instruct, 30b-thinking-2; semantic/linguistic selects 30b-instruct and 30b-instruct-lingu-1; prompt comparison selects 30b-instruct and 30b-improved. Original image files are retained.\n'
        'Statistics: every CSV mean and SD is verified against its stored observation arrays. Old comparison plots called the average within-label SD "Variance". The exported intervals retain those exact aggregate means and SDs, now explicitly defined. No square root conversion is appropriate.\n'
        'Sampling: label weights, repeated observations, unequal Think runs, and incomplete linguistic records are retained. Intervals are descriptive, not independent-sample uncertainty estimates. Prompt sample identifiers and source run arrays are checked. Raw generation metadata are absent.\n'
        'Diagnostic counts and denominators are reconstructed from case records and checked against archived summaries. Diagnostic subsets differ in size and do not cover all 510 evaluations per prompt.\n'
        'See captions.txt, validation_values.csv, diagnostic_counts.csv, and data_audit.json (source hashes and sampling structure).\n', encoding='utf-8')
    print(f'Exported four figures and verified data to {OUT}')


if __name__ == '__main__':
    main()
