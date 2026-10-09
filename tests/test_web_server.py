"""Viewer web server (src/web_ui/Web_Server.py).

Covers the Content-Security-Policy sent with HTML pages, port selection and
the server lifecycle, plugin static routes (including that no request path
reaches a file outside a route's directory) and named server-sent event
clients. tests/test_agent_composer.py checks that the policy is enforced in
Qt WebEngine; the same-origin guard is tested in tests/test_viewer_web_origin.py.
"""
import base64
import hashlib
import http.client
import os
from pathlib import Path
from queue import Queue
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from web_ui import Web_Server
from web_ui.Plugin_Manager import WebPluginRegistry
from web_ui.Web_Server import content_security_policy

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
PAGES = SRC_DIR / 'web_ui'


def directives(policy):
    parsed = {}
    for directive in policy.split(';'):
        name, *sources = directive.split()
        parsed[name] = sources
    return parsed


def sha256_source(text):
    return "'sha256-%s'" % base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


def use_private_session_directory(test):
    """Publish the Viewer session descriptors of ``test``'s servers to its own folder."""
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    environment = mock.patch.dict(os.environ, {'SSN_VIEWER_SESSION_DIR': directory.name})
    environment.start()
    test.addCleanup(environment.stop)


class FakeViewer:
    """What the server's routes and event stream need from a Viewer."""

    def get_initial_web_state(self):
        return {"base": True}


class PagePolicyTests(unittest.TestCase):
    def test_inline_scripts_are_allowed_by_hash_of_their_normalized_text(self):
        # Browsers hash a script's text after the HTML parser turns CRLF and CR
        # into LF; external scripts are covered by 'self' instead.
        body = (b'<script src="/x.js"></script><script>\r\nlet a = 1;\r\n</script>'
                b'<SCRIPT type="module">b()\r</SCRIPT >')
        script_src = directives(content_security_policy('page.html', body))['script-src']
        self.assertEqual(script_src, ["'self'", sha256_source('\nlet a = 1;\n'), sha256_source('b()\n')])

    def test_bundled_pages_cannot_run_injected_markup(self):
        for page in ('agent.html', 'meta.html', 'esmfold.html'):
            with self.subTest(page=page):
                policy = directives(content_security_policy(page, (PAGES / page).read_bytes()))
                script_src = policy['script-src']
                self.assertNotIn("'unsafe-inline'", script_src)
                self.assertEqual(sum(source.startswith("'sha256-") for source in script_src), 1)
                self.assertEqual(policy['default-src'], ["'self'"])
                self.assertEqual(policy['object-src'], ["'none'"])
                self.assertEqual(policy['base-uri'], ["'none'"])
                self.assertEqual(policy['frame-ancestors'], ["'none'"])

    def test_only_mol_star_may_evaluate_code_and_reach_public_servers(self):
        for page in ('agent.html', 'meta.html'):
            with self.subTest(page=page):
                policy = directives(content_security_policy(page, b''))
                self.assertNotIn("'unsafe-eval'", policy['script-src'])
                self.assertEqual(policy['connect-src'], ["'self'"])
                self.assertEqual(policy['img-src'], ["'self'", 'data:', 'blob:'])
        esmfold = directives(content_security_policy('esmfold.html', b''))
        self.assertIn("'unsafe-eval'", esmfold['script-src'])
        self.assertEqual(esmfold['connect-src'], ["'self'", 'https:', 'data:'])


class PagePolicyOverHTTPTests(unittest.TestCase):
    """The running server sends each HTML page the policy of the bytes it
    serves, and sends none with scripts or JSON."""

    def setUp(self):
        use_private_session_directory(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        files = Path(directory.name)
        (files / 'app.js').write_bytes(b'console.log(1);')
        (files / 'data.json').write_bytes(b'{"a": 1}')
        (files / 'page.html').write_bytes(b'<script>\r\nrun();\r\n</script>')
        viewer = FakeViewer()
        viewer.web_plugin_registry = WebPluginRegistry(viewer)
        viewer.web_plugin_registry.register_static_route('files', '/files/', str(files))
        self.server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)

    def get(self, path):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=10)
        try:
            connection.request('GET', path)
            response = connection.getresponse()
            return response, response.read()
        finally:
            connection.close()

    def test_bundled_pages_carry_the_policy_of_their_served_bytes(self):
        for page in ('agent.html', 'meta.html', 'esmfold.html'):
            with self.subTest(page=page):
                response, body = self.get(f'/{page}')
                self.assertEqual(response.status, 200)
                self.assertEqual(body, (PAGES / page).read_bytes())
                self.assertEqual(response.getheader('Content-Type'), 'text/html; charset=utf-8')
                self.assertEqual(
                    response.msg.get_all('Content-Security-Policy'),
                    [content_security_policy(page, body)],
                )
                script_src = directives(response.getheader('Content-Security-Policy'))['script-src']
                self.assertEqual("'unsafe-eval'" in script_src, page == 'esmfold.html')

    def test_the_bundled_chinese_font_is_served_for_chinese_pages(self):
        # A page served in Simplified Chinese names this file (Desktop_App.language_web_font_css).
        for name in ('NotoSansSC-Regular.ttf', 'NotoSansSC-Bold.ttf'):
            with self.subTest(name=name):
                response, body = self.get(f'/fonts/desktop/noto/NotoSansSC/{name}')
                self.assertEqual(response.status, 200)
                self.assertTrue(response.getheader('Content-Type').startswith('font/ttf'))
                self.assertEqual(body, (SRC_DIR / 'resources' / 'fonts' / 'desktop' / 'noto' / 'NotoSansSC' / name)
                                 .read_bytes())

    def test_route_served_pages_get_a_policy_too(self):
        response, _body = self.get('/files/page.html')
        self.assertEqual(response.status, 200)
        script_src = directives(response.getheader('Content-Security-Policy'))['script-src']
        self.assertEqual(script_src, ["'self'", sha256_source('\nrun();\n')])

    def test_scripts_and_json_carry_no_policy(self):
        for path, content_type in (
            ('/files/app.js', 'application/javascript'),
            ('/files/data.json', 'application/json'),
        ):
            with self.subTest(path=path):
                response, _body = self.get(path)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader('Content-Type'), f'{content_type}; charset=utf-8')
                self.assertIsNone(response.getheader('Content-Security-Policy'))


class ConcurrentWebServerTests(unittest.TestCase):
    def setUp(self):
        use_private_session_directory(self)

    def _close_server(self, server):
        Web_Server.stop_server(server)

    def test_server_uses_requested_available_port(self):
        server = Web_Server.start_server(SimpleNamespace(), preferred_port=0)
        self.addCleanup(self._close_server, server)
        self.assertGreater(server.server_address[1], 0)

    def test_occupied_preferred_port_falls_back_to_free_port(self):
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("localhost", 0))
        occupied.listen(1)
        self.addCleanup(occupied.close)
        occupied_port = occupied.getsockname()[1]

        server = Web_Server.start_server(SimpleNamespace(), preferred_port=occupied_port)
        self.addCleanup(self._close_server, server)
        self.assertNotEqual(server.server_address[1], occupied_port)
        self.assertGreater(server.server_address[1], 0)

    def test_two_viewer_servers_never_share_the_preferred_port(self):
        first = Web_Server.start_server(SimpleNamespace(), preferred_port=0)
        self.addCleanup(self._close_server, first)
        preferred_port = first.server_address[1]

        second = Web_Server.start_server(SimpleNamespace(), preferred_port=preferred_port)
        self.addCleanup(self._close_server, second)

        self.assertEqual(first.server_address[1], preferred_port)
        self.assertNotEqual(second.server_address[1], preferred_port)
        for server in (first, second):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_address[1]}/meta.html",
                timeout=2,
            ) as response:
                self.assertEqual(response.status, 200)

    def test_stop_is_idempotent_and_ensure_replaces_stopped_server(self):
        viewer = SimpleNamespace()
        first = Web_Server.start_server(viewer, preferred_port=0)
        first_port = first.server_address[1]
        self.assertTrue(Web_Server.is_running(first))

        Web_Server.stop_server(first)
        Web_Server.stop_server(first)
        self.assertFalse(Web_Server.is_running(first))
        with self.assertRaises(OSError):
            socket.create_connection((Web_Server.LOOPBACK_HOST, first_port), timeout=0.2)

        replacement = Web_Server.ensure_server(
            viewer, first, preferred_port=first_port
        )
        self.addCleanup(self._close_server, replacement)
        self.assertIsNot(replacement, first)
        self.assertEqual(replacement.inspection_session_id, first.inspection_session_id)
        self.assertTrue(Web_Server.is_running(replacement))

    def test_platform_address_reuse_policy(self):
        windows_server = SimpleNamespace(
            allow_reuse_address=True,
            socket=mock.Mock(),
        )
        with mock.patch.object(
            Web_Server.socket,
            "SO_EXCLUSIVEADDRUSE",
            12345,
            create=True,
        ):
            Web_Server._configure_address_reuse(
                windows_server,
                platform_name="nt",
            )
        self.assertFalse(windows_server.allow_reuse_address)
        windows_server.socket.setsockopt.assert_called_once_with(
            socket.SOL_SOCKET,
            12345,
            1,
        )

        posix_server = SimpleNamespace(
            allow_reuse_address=False,
            socket=mock.Mock(),
        )
        Web_Server._configure_address_reuse(
            posix_server,
            platform_name="posix",
        )
        self.assertTrue(posix_server.allow_reuse_address)
        posix_server.socket.setsockopt.assert_not_called()


class PluginRoutesAndEventClientsTests(unittest.TestCase):
    def setUp(self):
        use_private_session_directory(self)

    def test_server_uses_the_preconfigured_registry_route_mapping(self):
        viewer = FakeViewer()
        registry = WebPluginRegistry(viewer)
        viewer.web_plugin_registry = registry
        registry.register_static_route("alpha", "/alpha/", ".")
        server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, server)

        self.assertIs(server.static_routes, registry.static_routes)
        self.assertIn("/alpha/", server.static_routes)
        self.assertIn("/fonts/", server.static_routes)

    def test_server_tracks_named_event_clients_independently(self):
        viewer = FakeViewer()
        registry = WebPluginRegistry(viewer)
        viewer.web_plugin_registry = registry
        server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, server)
        meta_queue = Queue()
        esmfold_queue = Queue()
        viewer._web_ui_pending_opens = {"esmfold": float("inf")}

        server.register_event_queue(meta_queue, "meta")
        server.register_event_queue(esmfold_queue, "esmfold")
        self.assertTrue(server.has_event_client("meta"))
        self.assertTrue(server.has_event_client("esmfold"))
        self.assertNotIn("esmfold", viewer._web_ui_pending_opens)

        server.unregister_event_queue(esmfold_queue, "esmfold")
        self.assertTrue(server.has_event_client("meta"))
        self.assertFalse(server.has_event_client("esmfold"))
        server.unregister_event_queue(meta_queue, "meta")

    def test_event_client_url_labels_are_validated(self):
        self.assertEqual(
            Web_Server.event_client_from_path("/api/events?client=ESMFold"),
            "esmfold",
        )
        self.assertIsNone(Web_Server.event_client_from_path("/api/events"))
        self.assertIsNone(
            Web_Server.event_client_from_path("/api/events?client=esmfold%2Fother")
        )

    def test_named_sse_request_registers_and_releases_client(self):
        viewer = FakeViewer()
        registry = WebPluginRegistry(viewer)
        viewer.web_plugin_registry = registry
        server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, server)
        # Not localhost: Windows tries ::1 first, and the refused attempt
        # there costs about a second.
        url = (
            f"http://127.0.0.1:{server.server_address[1]}"
            "/api/events?client=esmfold"
        )

        response = urllib.request.urlopen(url, timeout=2)
        try:
            deadline = time.monotonic() + 10
            while not server.has_event_client("esmfold") and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(server.has_event_client("esmfold"))
        finally:
            response.close()

        # The server notices the closed stream when a keep-alive write fails;
        # it writes one every second.
        deadline = time.monotonic() + 10
        while server.has_event_client("esmfold") and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(server.has_event_client("esmfold"))

    def test_bundled_pages_identify_their_event_streams(self):
        expected_clients = {
            "esmfold.html": "esmfold",
            "meta.html": "meta",
            "agent.html": "agent",
        }
        web_ui_dir = SRC_DIR / "web_ui"
        for filename, client_id in expected_clients.items():
            with self.subTest(filename=filename):
                source = (web_ui_dir / filename).read_text(encoding="utf-8")
                self.assertIn(f"/api/events?client={client_id}", source)


class StaticRouteContainmentTests(unittest.TestCase):
    """No request path reaches a file outside a static route's directory.

    The routes serve the user's cache (/structures/) and the agent's
    model_card.json, which holds API keys (/agent_resource/). Paths go out
    exactly as written here: http.client neither normalizes nor quotes them.
    """

    SECRET = b'secret outside the route directory'

    def setUp(self):
        use_private_session_directory(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        files = root / 'files'
        files.mkdir()
        (files / 'ok.txt').write_bytes(b'served')
        # ESMFold keeps leading dots in names: the node "..x" folds into ..x.pdb.
        (files / '..x.pdb').write_bytes(b'structure')
        self.secret = root / 'secret.txt'
        self.secret.write_bytes(self.SECRET)
        # A sibling file and folder whose names extend the route directory's
        # name, so their paths start with its path.
        (root / 'files_secret.txt').write_bytes(self.SECRET)
        (root / 'files-private').mkdir()
        (root / 'files-private' / 'key.txt').write_bytes(self.SECRET)
        viewer = SimpleNamespace()
        viewer.web_plugin_registry = WebPluginRegistry(viewer)
        viewer.web_plugin_registry.register_static_route('files', '/files/', str(files))
        self.server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)

    def get(self, path):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=10)
        try:
            connection.request('GET', path)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def assert_replies(self, cases, withheld):
        for path, status in cases.items():
            with self.subTest(path=path):
                reply_status, body = self.get(path)
                self.assertNotIn(withheld, body)
                self.assertEqual(reply_status, status)

    def make_link(self, link, target, junction=False):
        """Point ``link`` at ``target``; it is removed before the fixture folder.

        Windows lets only administrators and Developer Mode create symbolic
        links, so the test is skipped without that privilege. Junctions need
        none.
        """
        if junction:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            try:
                os.symlink(target, link, target_is_directory=target.is_dir())
            except OSError as error:
                self.skipTest(f'this account cannot create symbolic links: {error}')
        # On Windows rmdir removes a folder link or junction, never its target.
        self.addCleanup(os.rmdir if os.name == 'nt' and target.is_dir() else os.unlink, link)

    def test_route_serves_files_inside_its_directory(self):
        self.assertEqual(self.get('/files/ok.txt'), (200, b'served'))

    def test_names_that_only_start_with_two_dots_are_served(self):
        # Only a whole ".." component climbs out of the directory.
        self.assertEqual(self.get('/files/..x.pdb'), (200, b'structure'))

    def test_relative_escapes_are_refused(self):
        cases = {
            '/files/../secret.txt': 403,
            # On POSIX a backslash is an ordinary filename character.
            '/files/..\\secret.txt': 403 if os.name == 'nt' else 404,
            '/files/../files_secret.txt': 403,
            '/files/../files-private/key.txt': 403,
        }
        if os.name == 'nt':
            drive = os.path.splitdrive(str(self.secret))[0]
            cases.update({
                # Drive-relative: os.path.join resolves these against the
                # route directory, so they name files beside it.
                '/files/' + drive + '../secret.txt': 403,
                '/files/' + drive + '../files_secret.txt': 403,
                # A drive is refused even where the join would stay inside.
                '/files/' + drive + 'secret.txt': 403,
            })
        self.assert_replies(cases, self.SECRET)

    def test_absolute_and_rooted_paths_are_refused(self):
        root = str(self.secret.parent)
        if not root.isascii() or ' ' in root:
            self.skipTest('http.client cannot send this temporary path unquoted')
        cases = {}
        for name in ('secret.txt', 'files_secret.txt', 'files-private/key.txt'):
            path = os.path.join(root, *name.split('/'))
            # Rooted without a drive, which Python 3.13's ntpath.isabs no
            # longer counts as absolute.
            rooted = os.path.splitdrive(path)[1]
            cases['/files/' + path.replace(os.sep, '/')] = 403
            cases['/files/' + rooted.replace(os.sep, '/')] = 403
            # On POSIX a backslash is an ordinary filename character.
            cases['/files/' + rooted.replace(os.sep, '\\')] = 403 if os.name == 'nt' else 404
        self.assert_replies(cases, self.SECRET)

    @unittest.skipUnless(os.name == 'nt', 'Windows device names')
    def test_device_names_are_refused(self):
        # Windows opens NUL in any folder as the \\.\nul device, which lies on
        # no drive.
        self.assert_replies({'/files/nul': 403, '/files/sub/NUL': 403}, self.SECRET)

    def test_links_out_of_the_directory_are_refused(self):
        root = self.secret.parent
        self.make_link(root / 'files' / 'file-link.txt', root / 'files_secret.txt')
        self.make_link(root / 'files' / 'folder-link', root / 'files-private')
        self.assert_replies({
            '/files/file-link.txt': 403,
            '/files/folder-link/key.txt': 403,
        }, self.SECRET)

    @unittest.skipUnless(os.name == 'nt', 'Windows junctions')
    def test_junctions_out_of_the_directory_are_refused(self):
        root = self.secret.parent
        self.make_link(root / 'files' / 'junction', root / 'files-private', junction=True)
        self.assert_replies({'/files/junction/key.txt': 403}, self.SECRET)

    def test_a_route_directory_reached_through_a_link_is_served(self):
        # Such as a cache folder moved to another drive and linked back.
        root = self.secret.parent
        self.make_link(root / 'linked', root / 'files', junction=os.name == 'nt')
        self.server.static_routes['/linked/'] = str(root / 'linked')
        self.assertEqual(self.get('/linked/ok.txt'), (200, b'served'))

    def test_links_that_stay_inside_the_directory_are_served(self):
        files = self.secret.parent / 'files'
        self.make_link(files / 'alias.txt', files / 'ok.txt')
        self.assertEqual(self.get('/files/alias.txt'), (200, b'served'))

    def test_names_are_not_percent_decoded(self):
        # %2e%2e is looked up as a literal name inside the route directory.
        self.assert_replies({'/files/%2e%2e/secret.txt': 404, '/files/': 404}, self.SECRET)

    def test_bundled_pages_route_stays_inside_web_ui(self):
        config = (SRC_DIR / 'EMAPSSN_Config.py').read_bytes()[:400]
        self.assert_replies({
            '/../EMAPSSN_Config.py': 403,
            # On POSIX a backslash is an ordinary filename character.
            '/\\..\\EMAPSSN_Config.py': 403 if os.name == 'nt' else 404,
        }, config)

    def test_bundled_pages_route_applies_the_same_rules(self):
        # Serve the bundled pages from the route directory, beside which the
        # secrets lie.
        root = self.secret.parent
        drive, rooted = os.path.splitdrive(str(root))
        cases = {
            '/ok.txt': 200,
            '/..x.pdb': 200,
            '/../files_secret.txt': 403,
        }
        if os.name == 'nt':
            cases.update({
                '/' + drive + '../files_secret.txt': 403,
                '/' + drive + '../files-private/key.txt': 403,
            })
            if rooted.isascii() and ' ' not in rooted:
                # In backslashes: http.server collapses a leading "//".
                cases.update({
                    '/' + rooted + '\\files_secret.txt': 403,
                    '/' + rooted + '\\files-private\\key.txt': 403,
                })
        with mock.patch.object(Web_Server, 'BASE_DIR', str(root / 'files')):
            self.assert_replies(cases, self.SECRET)


if __name__ == '__main__':
    unittest.main()
