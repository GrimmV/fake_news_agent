"""Report the exact exported values used in regenerated Figures 7.2–7.5.

Examples:
    python report_chapter7_values.py
    python report_chapter7_values.py --figure 7.4
    python report_chapter7_values.py --format json --output figures/chapter7/exact_values.json

Uses only the standard library. Reads the figure generator's CSV exports and
audit, without regenerating figures or rounding their stored decimal values.
"""
import argparse
import csv
import json
from pathlib import Path


DATA = Path(__file__).resolve().parent / 'figures' / 'chapter7'
CONFIGURATIONS = {
    '7.2': [('4B', '4b'), ('8B', '8b'), ('30B Instruct', '30b-instruct'),
            ('30B Think', '30b-thinking-2')],
    '7.3': [('Semantic', '30b-instruct'), ('Linguistic', '30b-instruct-lingu-1')],
    '7.5': [('Original prompt', '30b-instruct'), ('Revised prompt', '30b-improved')],
}
METRICS = [
    ('Source-near and intermediate transformations', [
        ('xai_description_truthfulness', 'Source-to-description truthfulness'),
        ('technical_xai_clarity', 'Technical-description clarity'),
        ('layman_xai_truthfulness', 'Technical-to-layperson truthfulness'),
    ]),
    ('High-level assessment and rationale', [
        ('label_correlation', 'Dataset-label alignment'),
        ('technical_assessment_clarity', 'Assessment-rationale clarity'),
        ('short_assessment_truthfulness', 'Technical-to-short-rationale truthfulness'),
    ]),
]


def read_csv(filename):
    with (DATA / filename).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def build_report(figures):
    values = read_csv('validation_values.csv')
    lookup = {(r['configuration'], r['metric']): r for r in values}
    if len(lookup) != len(values):
        raise ValueError('Duplicate configuration/metric rows in validation_values.csv')
    diagnoses = read_csv('diagnostic_counts.csv')
    audit = json.loads((DATA / 'data_audit.json').read_text(encoding='utf-8'))
    report = {
        'source': str(DATA),
        'precision': 'Decimal strings are reproduced verbatim from the figure exports; no additional rounding. These preserve stored floating-point precision, not infinite mathematical precision.',
        'dispersion': audit['dispersion'],
        'interval': 'mean +/- mean_within_label_sample_sd; endpoints are not clipped to [0, 1].',
        'diagnoses': 'Primary is the main diagnosis; secondary diagnoses may co-occur. Percentages are supplementary and use each prompt\'s divergent denominator, not all evaluated assessments.',
        'figures': {},
    }
    for figure in figures:
        if figure == '7.4':
            panels = []
            for label, key in [('Original prompt', 'initial'), ('Revised prompt', 'improved')]:
                totals = next(d for d in audit['diagnostics'] if d['name'] == key)
                rows = [dict(r) for r in diagnoses if r['prompt'] == label]
                if not rows or any(int(r['divergent_denominator']) != totals['total'] for r in rows):
                    raise ValueError(f'Diagnostic denominator mismatch: {label}')
                if sum(int(r['count']) for r in rows if r['diagnosis'] == 'primary') != totals['total']:
                    raise ValueError(f'Primary counts do not sum to the denominator: {label}')
                panels.append({'prompt': label, 'divergent_assessments': totals['total'],
                               'evaluated_assessments': totals['evaluated'], 'values': rows})
            report['figures'][figure] = panels
        else:
            rows = []
            for column, metrics in METRICS:
                for metric, title in metrics:
                    for label, config in CONFIGURATIONS[figure]:
                        rows.append({'column': column, 'metric_label': title,
                                     'condition': label, **lookup[config, metric]})
            report['figures'][figure] = rows
    return report


def render_text(report):
    lines = ['Exact values in Figures 7.2–7.5', f"Source: {report['source']}"]
    lines.extend(report[k] for k in ['precision', 'dispersion', 'interval', 'diagnoses'])
    for figure, entries in report['figures'].items():
        lines.extend(['', f'Figure {figure}', '=' * 72])
        if figure == '7.4':
            for panel in entries:
                lines.extend(['', f"{panel['prompt']}: {panel['divergent_assessments']} divergent / {panel['evaluated_assessments']} evaluated assessments",
                              'Category\tDiagnosis\tCount\tDivergent denominator\tPercent of divergent (supplementary)'])
                lines.extend('\t'.join(r[k] for k in ['category', 'diagnosis', 'count',
                                                    'divergent_denominator', 'percent_of_divergent'])
                             for r in panel['values'])
        else:
            previous = None
            for row in entries:
                heading = (row['column'], row['metric_label'])
                if heading != previous:
                    lines.extend(['', ' / '.join(heading),
                                  'Condition\tMean\tMean within-label sample SD\tLower endpoint\tUpper endpoint\tN observations'])
                    previous = heading
                lines.append('\t'.join(row[k] for k in ['condition', 'mean',
                                                        'mean_within_label_sample_sd', 'lower', 'upper', 'n']))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--figure', nargs='+', choices=['7.2', '7.3', '7.4', '7.5'],
                        default=['7.2', '7.3', '7.4', '7.5'], help='Figures to report (default: all).')
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    parser.add_argument('--output', type=Path, help='Write a UTF-8 file instead of printing to the terminal.')
    args = parser.parse_args()
    try:
        report = build_report(args.figure)
    except (OSError, KeyError, ValueError, StopIteration) as exc:
        parser.error(f'Cannot read figure data: {exc}. Run regenerate_chapter7_figures.py to refresh the exports.')
    output = json.dumps(report, indent=2, ensure_ascii=False) + '\n' if args.format == 'json' else render_text(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
