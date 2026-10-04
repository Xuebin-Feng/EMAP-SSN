"""Content-Security-Policy that Web_Server.serve_file sends with HTML pages.

tests/test_agent_composer.py checks that the policy is enforced in Qt WebEngine.
"""
import base64
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from web_ui.Web_Server import content_security_policy

PAGES = Path(__file__).resolve().parents[1] / 'src' / 'web_ui'


def directives(policy):
    parsed = {}
    for directive in policy.split(';'):
        name, *sources = directive.split()
        parsed[name] = sources
    return parsed


def sha256_source(text):
    return "'sha256-%s'" % base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


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


if __name__ == '__main__':
    unittest.main()
