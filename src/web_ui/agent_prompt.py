# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""The Agent's system prompt: src/resources/agent/system_prompt.md, with what depends on the code filled in.

The file is written by hand, except for the list of commands and the lists of how a
command in an Agent reply is handled (run at once, wait for the user's approval, refuse).

* The numbered list of commands is made from the command modules when the prompt is
  loaded: each module declares its own AGENT_PROMPT, its text, and AGENT_PROMPT_ORDER,
  its place in the list (desktop.Command_Metadata). A command offered to the Agent that
  declares no text gets a short item made from its catalog summary and syntax, so a
  plugin added to src/commands reaches the Agent with no change here or in the file.
* The review lists are written from each module's AGENT_POLICY, so the instructions
  cannot disagree with the policy the Viewer applies.
"""
import re

from desktop.Command_Metadata import (
    APPROVE, REFUSE, RUN, agent_policy_summary, agent_prompt_items, unlisted_agent_policy)

COMMAND_ITEMS = '%%COMMAND_ITEMS%%'
RUN_AT_ONCE = '%%RUN_AT_ONCE%%'
WAIT_FOR_USER = '%%WAIT_FOR_USER%%'
REFUSED = '%%REFUSED%%'
BY_FORM = '%%BY_FORM%%'
UNLISTED = '%%UNLISTED%%'
PLACEHOLDERS = (COMMAND_ITEMS, RUN_AT_ONCE, WAIT_FOR_USER, REFUSED, BY_FORM, UNLISTED)

_PHRASES = {RUN: 'runs at once', APPROVE: 'waits for the user to press Run', REFUSE: 'is refused'}


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


def generated_item(name, path):
    """A short item for a command whose module declares no AGENT_PROMPT: its syntax and its catalog summary.

    The catalog's syntax is the usage lines of the module's own help, so each form starts with the command's name.
    """
    from desktop.Viewer_Inspection import command_catalog_entry
    try:
        found = command_catalog_entry(path)
    except (OSError, SyntaxError, ValueError):
        found = None
    syntax = (found or {}).get('syntax') or []
    lines = [f'`{syntax[0] if syntax else name}`', f"- {(found or {}).get('summary') or f'The {name} command.'}"]
    if len(syntax) > 1:
        lines.append(f"- Other forms: {', '.join(f'`{form}`' for form in syntax[1:])}.")
    return '\n'.join(lines)


def numbered_item(number, text):
    """text as item ``number`` of the list: the number before its first line, the others indented to stand under it."""
    lead = f'{number}. '
    pad = ' ' * len(lead)
    first, *rest = text.strip().split('\n')
    return lead + first + ''.join('\n' + (pad + line if line.strip() else '') for line in rest)


def command_items():
    """The numbered list of the commands a reply may use, from the command modules, separated by blank lines."""
    items = []
    for number, item in enumerate(agent_prompt_items(), 1):
        items.append(numbered_item(number, item.text or generated_item(item.name, item.path)))
    return '\n\n'.join(items)


def render_system_prompt(template):
    """template with the command list and the policy lists filled in; each placeholder must occur exactly once."""
    plain, by_form = agent_policy_summary()
    values = {
        COMMAND_ITEMS: command_items(),
        RUN_AT_ONCE: _spans(plain[RUN]),
        WAIT_FOR_USER: _spans(plain[APPROVE]),
        REFUSED: _spans(plain[REFUSE]),
        BY_FORM: by_form_lines(by_form) or '  - none',
        UNLISTED: _PHRASES[unlisted_agent_policy()],
    }
    for token in values:
        if template.count(token) != 1:
            raise ValueError(f'{token} must occur exactly once in the system prompt, not {template.count(token)} times.')
    # One pass, so that what is put in (a module's own text) is never read as a placeholder.
    return re.sub('|'.join(re.escape(token) for token in values), lambda found: values[found.group(0)], template)
