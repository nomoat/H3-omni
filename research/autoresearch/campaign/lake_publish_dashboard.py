"""Save the H3 dashboard and verify its resolved data after lake ingestion."""
import argparse
import copy
from datetime import datetime, timezone
import html
from pathlib import Path
import re

from lake_dashboard import DASHBOARD_NAME, build_spec
from lake_export import canonical, client_module, read, write


def preview(root, spec):
    """Use AutoGrad's existing viewer with exact local rows, without lake writes."""
    source = Path('/home/homo/model/srp/AutoGrad/services/data-lake/src')
    payload = read(read(root / 'lake/latest.json')['payload'])
    snapshot = copy.deepcopy(spec)
    snapshot['title'] += ' — local snapshot; lake publication pending'
    snapshot['refresh_seconds'] = 0
    data = []
    for panel in snapshot['panels']:
        rows = []
        for item in panel['data']:
            stream = item['stream']
            rows.extend(dict(row, _stream=stream, _experiment=payload['experiment_id'])
                        for row in payload['streams'].get(stream, [])[-item.get('limit', 5000):])
        data.append(rows)
    resolved = {'spec': snapshot, 'data': data}
    page = (source / 'dashboard.html').read_text()
    page = page.replace('__TITLE__', html.escape(snapshot['title']))
    page = page.replace('__CSS__', (source / 'dashboard.css').read_text())
    page = page.replace('__JS__', (source / 'dashboard.js').read_text())
    page = page.replace('__DATA__', canonical(resolved).replace('</', '<\\/'))
    bundles = source.parents[2] / 'apps/web/public/monitor'
    def inline_script(match):
        filename = match.group(1)
        return '<script>' + (bundles / filename).read_text().replace('</script', '<\\/script') + '</script>'
    page = re.sub(r'<script src="https://cdn\.jsdelivr\.net/npm/[^\"]+/build/(vega(?:-lite|-embed)?\.min\.js)"></script>',
                  inline_script, page)
    (root / 'lake/dashboard-preview.html').write_text(page)
    write(root / 'lake/dashboard-preview-data.json', resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    spec = build_spec()
    output = root / 'lake/dashboard.json'
    write(output, spec)
    preview(root, spec)
    if args.prepare_only:
        print(output)
        return
    module = client_module()
    client = module.LakeClient(timeout=60)
    payload = read(read(root / 'lake/latest.json')['payload'])
    # Publishing a graph without its expected underlying rows is not success.
    receipt = {'dashboard': DASHBOARD_NAME, 'verified': False,
               'url': 'https://autograd.run/monitor/#dashboard=' + DASHBOARD_NAME}
    try:
        receipt['input_verification'] = client.verify(payload['experiment_id'], payload['streams'])
        route = '/v1/dashboards/' + DASHBOARD_NAME
        try:
            previous = client.request('GET', route)
        except module.LakeHTTPError as error:
            if error.status != 404:
                raise
            previous = None
        if previous != spec:
            if previous is not None:
                write(root / 'lake/dashboard-previous.json', previous)
            client.request('PUT', route, spec)
        saved = client.request('GET', route)
        if saved != spec:
            raise ValueError('Saved dashboard differs from reviewed spec')
        resolved = client.request('GET', route + '/resolved')
        write(root / 'lake/dashboard-resolved.json', resolved)
        if len(resolved.get('data', [])) != len(spec['panels']):
            raise ValueError('Dashboard did not resolve every panel')
        expected_overview = payload['streams']['search/overview'][-1]
        overview = resolved['data'][0]
        if len(overview) != 1 or any(overview[0].get(key) != expected_overview.get(key)
                for key in ('status', 'completed', 'total', 'remaining', 'active')):
            raise ValueError('Dashboard progress differs from verified experiment state')
        rows = [row for panel in resolved['data'] for row in panel]
        expected_ids = {row['event_id'] for stream in ('metrics/cold_samples', 'packages/catalog')
                        for row in payload['streams'].get(stream, [])}
        actual_ids = {row.get('event_id') for row in rows}
        if not expected_ids <= actual_ids:
            raise ValueError('Dashboard is missing raw samples or package download records')
        receipt.update(verified=True, panels=len(spec['panels']),
                       panel_row_counts=[len(panel) for panel in resolved['data']],
                       browser_visual_review=False,
                       validation='Saved spec readback, every panel resolved, progress and sample/package identities checked')
    except Exception as error:
        receipt['error'] = module.redact_error(str(error), client.token)
        raise RuntimeError(receipt['error']) from None
    finally:
        receipt['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(root / 'lake/dashboard-receipt.json', receipt)
    print(receipt['url'])


if __name__ == '__main__':
    main()
