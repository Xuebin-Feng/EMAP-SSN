# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""The Agent's system prompt: src/resources/agent/system_prompt.md, with what depends on the code filled in.

The file is written by hand, except for the lists of how a command in an Agent
reply is handled (run at once, wait for the user's approval, refuse). Those are
written from desktop.Command_Metadata.AGENT_POLICY when the prompt is loaded, so
the instructions cannot disagree with the policy the Viewer applies, and a
command added to a plugin needs no change here.
"""
from desktop.Command_Metadata import agent_policy_summary, unlisted_agent_policy

RUN_AT_ONCE = '%%RUN_AT_ONCE%%'
WAIT_FOR_USER = '%%WAIT_FOR_USER%%'
REFUSED = '%%REFUSED%%'
BY_FORM = '%%BY_FORM%%'
UNLISTED = '%%UNLISTED%%'
PLACEHOLDERS = (RUN_AT_ONCE, WAIT_FOR_USER, REFUSED, BY_FORM, UNLISTED)

_PHRASES = {'run': 'runs at once', 'approve': 'waits for the user to press Run', 'refuse': 'is refused'}


def _spans(words):
    return ', '.join(f'`{word}`' for word in words) or 'none'


def _alternatives(words):
    """words as code spans: `a`, `b` or `c`."""
    spans = [f'`{word}`' for word in words]
    return spans[0] if len(spans) == 1 else ', '.join(spans[:-1]) + ' or ' + spans[-1]


def _forms(words):
    """What the first word after a command is: no argument at all, or one of words."""
    named = [word for word in words if word]
    parts = (['no argument'] if '' in words else []) + ([f'the first word {_alternatives(named)}'] if named else [])
    return ', or with '.join(parts)


def by_form_lines(by_form):
    """One bullet for each command whose handling depends on its first argument."""
    lines = []
    for name, default, exceptions in by_form:
        clauses = []
        for kind in dict.fromkeys(exceptions.values()):
            words = [word for word, other in exceptions.items() if other == kind]
            clauses.append(f'{_PHRASES[kind]} only with {_forms(words)}')
        lines.append(f"  - `{name}` {'; '.join(clauses)}; any other form {_PHRASES[default]}.")
    return '\n'.join(lines)


def render_system_prompt(template):
    """template with the policy lists filled in; each placeholder must occur exactly once."""
    plain, by_form = agent_policy_summary()
    values = {
        RUN_AT_ONCE: _spans(plain['run']),
        WAIT_FOR_USER: _spans(plain['approve']),
        REFUSED: _spans(plain['refuse']),
        BY_FORM: by_form_lines(by_form) or '  - none',
        UNLISTED: _PHRASES[unlisted_agent_policy()],
    }
    for token, value in values.items():
        if template.count(token) != 1:
            raise ValueError(f'{token} must occur exactly once in the system prompt, not {template.count(token)} times.')
        template = template.replace(token, value)
    return template
