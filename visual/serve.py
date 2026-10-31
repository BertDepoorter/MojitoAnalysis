"""Local server for the Mojito EMRI explorer: static files + live FEW trajectories + lazy TD windows.

Usage: python serve.py [--port 8765] [--no-browser]
"""
import argparse
import json
import os
import threading
import webbrowser
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from mojito import MojitoL1File

import build

_files, _lock = {}, threading.Lock()


def brick(k):
    with _lock:
        if k not in _files:
            path = build.find_brick(k)
            if path is None:
                raise ValueError(f'no brick for source {k}')
            _files[k] = MojitoL1File(path)
        return _files[k]


@lru_cache(maxsize=256)
def trajectory(M, mu, a, p0, e0, x0, T, harm):
    from few.utils.constants import YRSID_SI
    return build.tolist(build.track(M, mu, a, p0, e0, x0, T * YRSID_SI, harmonics=harm))


def api_trajectory(q):
    g = lambda key, default=None: float(q[key][0]) if key in q else default
    M, mu, a, p0, e0, x0, T = g('M'), g('mu'), g('a'), g('p0'), g('e0'), g('x0', 1.0), g('T', 2.0)
    if None in (M, mu, a, p0, e0):
        raise ValueError('M, mu, a, p0, e0 are required')
    if not (0 <= e0 < 0.9):
        raise ValueError('need 0 <= e0 < 0.9')
    if not (0 <= a < 0.999):
        raise ValueError('need 0 <= a < 0.999 (positive primary spin; prograde/retrograde via x0)')
    if x0 not in (-1.0, 1.0):
        raise ValueError('x0 = cos(iota) must be +1 or -1 (equatorial model)')
    if not (M > 0 and 0 < mu < M and 0 < T <= 10):
        raise ValueError('need M > mu > 0 and 0 < T <= 10 yr')
    from few.utils.geodesic import get_separatrix
    p_sep = float(get_separatrix(a, e0, x0))
    if p0 <= p_sep + 0.1:
        raise ValueError(f'p0 must exceed p_sep + 0.1 = {p_sep + 0.1:.3f}')
    harm = tuple(tuple(int(v) for v in h.split(':')) for h in q.get('harm', ['2:0:0,2:0:1'])[0].split(','))
    r = lambda v: float(f'{v:.10g}')
    return trajectory(r(M), r(mu), r(a), r(p0), r(e0), x0, r(T), harm)


def api_td(q):
    k, ch = int(q['k'][0]), q.get('ch', ['X'])[0].upper()
    if ch not in 'XYZAET' or len(ch) != 1:
        raise ValueError('ch must be one of X Y Z A E T')
    f = brick(k)
    ts = f.tdis.time_sampling
    t0, t1 = float(q['t0'][0]), float(q['t1'][0])
    sl = ts.slice_between(ts.t0 + max(t0, 0.0), ts.t0 + t1)
    d = getattr(f.tdis, f'{ch.lower()}2')[sl] / f.laser_frequency
    t = ts.t()[sl] - ts.t0
    if len(d) > 20000:
        lo, hi = build.envelope(d, 10000)
        m = -(-len(d) // 10000)
        return {'mode': 'envelope', 't': (t[0] + (np.arange(10000) + 0.5) * m * ts.dt).tolist(),
                'min': lo.tolist(), 'max': hi.tolist()}
    return {'mode': 'raw', 't': t.tolist(), 'y': d.tolist()}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=build.HERE, **kw)

    def do_GET(self):
        url = urlparse(self.path)
        routes = {'/api/trajectory': api_trajectory, '/api/td': api_td}
        if url.path not in routes:
            return super().do_GET()
        try:
            code, body = 200, routes[url.path](parse_qs(url.query))
        except (ValueError, KeyError) as exc:
            code, body = 400, {'error': str(exc)}
        except Exception as exc:
            code, body = 500, {'error': repr(exc)}
            self.log_error('%s', repr(exc))
        data = json.dumps(build.tolist(body), allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Serving Mojito EMRI explorer at {url}  (Ctrl+C to stop)')
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
