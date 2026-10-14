"""Run visualizer server. From the repo root:

    python visualizer/serve.py                 # reads ./results, opens the browser
    python visualizer/serve.py --port 9000 --no-browser
    python visualizer/serve.py --results /path/to/results

Lists every run in the results folder that has a run_info.json (written at the end of each
test run) and serves the page plus each run's report.json. Uses only the standard library and
listens on localhost only unless --host says otherwise.

On a remote machine (SSH, or Linux without a display) it does not open a browser; it prints how
to forward the port to your own computer instead.
"""
import argparse
import getpass
import gzip
import json
import os
import socket
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, 'index.html')


def list_runs(results):
    runs = []
    if not os.path.isdir(results):
        return runs
    for name in os.listdir(results):
        info_path = os.path.join(results, name, 'run_info.json')
        if not os.path.isfile(info_path):
            continue
        try:
            with open(info_path) as f:
                runs.append(json.load(f))
        except (OSError, ValueError) as e:
            print('skipping {}: {}'.format(info_path, e))
    runs.sort(key=lambda r: r.get('created', ''), reverse=True)
    return runs


def make_handler(results):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, content_type, status=200):
            if 'gzip' in self.headers.get('Accept-Encoding', '') and len(body) > 1024:
                body = gzip.compress(body, compresslevel=5)
                gz = True
            else:
                gz = False
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            if gz:
                self.send_header('Content-Encoding', 'gzip')
            self.end_headers()
            self.wfile.write(body)

        def _not_found(self, what):
            self._send(json.dumps({'error': what}).encode(), 'application/json', 404)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ('/', '/index.html'):
                with open(PAGE, 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif path == '/api/runs':
                self._send(json.dumps(list_runs(results)).encode(), 'application/json')
            elif path.startswith('/api/run/'):
                run_id = unquote(path[len('/api/run/'):])
                # run ids are folder names directly under results/, nothing else is served
                if not run_id or run_id != os.path.basename(run_id) or run_id.startswith('.'):
                    return self._not_found('bad run id')
                report = os.path.join(results, run_id, 'report.json')
                if not os.path.isfile(report):
                    return self._not_found('no report.json for this run')
                with open(report, 'rb') as f:
                    self._send(f.read(), 'application/json')
            else:
                self._not_found('not found')

        def log_message(self, fmt, *args):
            pass

    return Handler


def is_remote_session():
    """True over SSH or on a Linux machine without a display, where opening a browser makes no sense."""
    if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'):
        return True
    return sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def ssh_host():
    """The address you used to reach this machine: SSH records it as the third field of SSH_CONNECTION."""
    parts = os.environ.get('SSH_CONNECTION', '').split()
    return parts[2] if len(parts) >= 3 else socket.gethostname()


def main():
    parser = argparse.ArgumentParser(description='Serve the run visualizer')
    parser.add_argument('--results', default='results', help='folder holding one subfolder per run')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1',
                        help='address to listen on; 0.0.0.0 makes the page reachable by anyone on the network')
    parser.add_argument('--no-browser', action='store_true', help='do not open a browser tab')
    args = parser.parse_args()

    results = os.path.abspath(args.results)
    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer((args.host, port), make_handler(results))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit('No free port between {} and {}'.format(args.port, args.port + 19))

    port = server.server_address[1]
    url = 'http://127.0.0.1:{}/'.format(port)
    print('Runs from {} ({} found)'.format(results, len(list_runs(results))), flush=True)
    if is_remote_session():
        print('Visualizer running on {} port {}  (Ctrl+C to stop)'.format(socket.gethostname(), port), flush=True)
        if args.host == '127.0.0.1':
            print('\nThis looks like a remote machine. To open the page on your own computer, either:\n'
                  '  - let your editor forward port {p} (VS Code: Ports panel; PyCharm/Gateway: Forwarded Ports), or\n'
                  '  - run this on your own computer:  ssh -N -L {p}:127.0.0.1:{p} {u}@{h}\n'
                  'then open {url}\n'.format(p=port, u=getpass.getuser(), h=ssh_host(), url=url), flush=True)
        else:
            print('Listening on {}:{}; open http://<this machine>:{}/ from your browser.'.format(args.host, port, port), flush=True)
    else:
        print('Visualizer at {}  (Ctrl+C to stop)'.format(url), flush=True)
        if not args.no_browser:
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()