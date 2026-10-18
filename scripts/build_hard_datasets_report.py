"""Render hard-task results offline, optionally recomputing from saved evidence."""
import argparse
from pathlib import Path
from _hard_report import collect, read, render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, help='Recompute from this saved completed campaign, offline')
    parser.add_argument('--summary', type=Path, default=Path('reports/v2/hard_datasets_results.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/v2'))
    parser.add_argument('--update-readme', action='store_true')
    args = parser.parse_args(argv)
    payload = collect(args.campaign_root) if args.campaign_root else read(args.summary)
    render(payload, args.output_dir, Path('README.md') if args.update_readme else None)
    print(f"Hard-task reports rebuilt offline: {len(payload['conditions'])}/34 included conditions complete")


if __name__ == '__main__':
    main()
