# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Image validation shared by the upload endpoint and agent messages."""
import base64
import io
import json
import warnings
import xml.etree.ElementTree as ET

from PIL import Image

from utilities.Localization import Message

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ATTACHMENTS = 10
MAX_DIMENSION = 1600


def inspect_source(raw):
    """Reject animation before the browser rasterizes the original image."""
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(Message('Each image must be nonempty and at most 20 MiB.'))
    # SVG is decoded by the browser, not Pillow. Only self-contained static SVG
    # is accepted: scripts and embedded resources could animate indirectly.
    if raw.lstrip().startswith((b'<', b'\xef\xbb\xbf', b'\xff\xfe', b'\xfe\xff')):
        try:
            root = ET.fromstring(raw)
            if root.tag.split('}')[-1] != 'svg':
                raise ValueError(Message('Only image files are accepted.'))
            for element in root.iter():
                tag = element.tag.split('}')[-1].lower()
                if tag in {'animate', 'animatemotion', 'animatetransform', 'set',
                           'script', 'foreignobject', 'image'}:
                    raise ValueError(Message('SVG must be static and self-contained, without animation or embedded images.'))
                if tag == 'style' and any(token in (element.text or '').lower()
                                          for token in ('animation', 'transition', '@', '\\', 'url(')):
                    raise ValueError(Message('Animated or external SVG styles are not accepted.'))
                for key, value in element.attrib.items():
                    key = key.split('}')[-1].lower()
                    if (key.startswith('on') or any(token in value.lower() for token in ('animation', 'transition', '\\'))
                            or (key == 'href' and not value.startswith('#'))):
                        raise ValueError(Message('SVG must be static and self-contained.'))
            return {'mime_type': 'image/svg+xml'}
        except ET.ParseError as error:
            raise ValueError(Message('Invalid or unsupported image file.')) from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as img:
                if getattr(img, 'n_frames', 1) != 1 or getattr(img, 'is_animated', False):
                    raise ValueError(Message('Animated and multi-frame images are not accepted.'))
                if img.format == 'ICO' and len(img.ico.sizes()) > 1:
                    raise ValueError(Message('Multi-frame icons are not accepted.'))
                mime = Image.MIME.get(img.format, 'application/octet-stream')
                img.load()
                return {'mime_type': mime}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(Message('Invalid or unsupported image file; only decodable static images are accepted.')) from error


def validate_attachments(attachments):
    if attachments is None:
        return []
    if not isinstance(attachments, list) or len(attachments) > MAX_ATTACHMENTS:
        raise ValueError(Message('A message can contain at most 10 image attachments.'))
    result = []
    prefix = 'data:image/png;base64,'
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError(Message('Invalid image attachment.'))
        url = attachment.get('data_url', '')
        if not isinstance(url, str) or not url.startswith(prefix) or len(url) > MAX_IMAGE_BYTES * 4 // 3 + 100:
            raise ValueError(Message('Attachments must be PNG images of at most 20 MiB.'))
        try:
            raw = base64.b64decode(url[len(prefix):], validate=True)
            info = inspect_source(raw)
            with Image.open(io.BytesIO(raw)) as img:
                if info['mime_type'] != 'image/png' or max(img.size) > MAX_DIMENSION:
                    raise ValueError(Message('Attachments must be PNG images no larger than 1600 pixels per side.'))
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(Message('Invalid PNG attachment.')) from error
        result.append({'name': str(attachment.get('name') or 'Image')[:255], 'data_url': url})
    return result


def message_content(text, attachments=None):
    """Keep legacy text-only messages unchanged."""
    if not attachments:
        return text
    return ([{'type': 'text', 'text': text}] if text else []) + [
        {'type': 'image_url', 'image_url': {'url': item['data_url']}}
        for item in attachments
    ]


# What each request re-sends of the conversation. The saved history stays whole, for the page to show.
HISTORY_MAX_TURNS = 10
HISTORY_TOKEN_BUDGET = 8000   # tokens, estimated as characters / CHARS_PER_TOKEN
CHARS_PER_TOKEN = 4
IMAGE_TOKENS = 1000           # what an image counts as: its data URL is not text the model reads
SUMMARY_MAX_COMMANDS = 20     # commands listed in the summary of an older turn's results
SUMMARY_MESSAGE_CHARS = 160   # characters of a command's own message kept in that summary
_OUTPUT_MARKER = '\n\nTerminal output:'


def estimate_tokens(text):
    return -(-len(text) // CHARS_PER_TOKEN)


def _command_results(terminal_output):
    """One line for each command of a turn's saved command output (its JSON), or [] when it has none."""
    try:
        commands = json.loads(terminal_output)['result']['commands']
        lines = []
        for command in commands[:SUMMARY_MAX_COMMANDS]:
            line = f"- {command['command']}: {command['status']}"
            messages = [m['text'] for m in command.get('messages') or [] if m.get('text')]
            if messages and command['status'] != 'succeeded':
                line += f" ({messages[-1][:SUMMARY_MESSAGE_CHARS].strip()})"
            lines.append(line)
        if len(commands) > SUMMARY_MAX_COMMANDS:
            lines.append(f'- and {len(commands) - SUMMARY_MAX_COMMANDS} more commands')
        return lines
    except (TypeError, ValueError, KeyError, AttributeError, IndexError):
        return []


def _summarized(assistant):
    """An assistant message of the history as its explanation and a short summary of what its commands did.

    The command output and the node headers it can hold stay out: an older turn's results are not worth
    their size, and they are the part of the history that holds the most of the user's data.
    """
    explanation = assistant.get('explanation')
    if explanation is None:  # Saved before the fields: its text, without the output block it ends with.
        return str(assistant.get('content', '')).split(_OUTPUT_MARKER, 1)[0]
    results = _command_results(assistant.get('terminal_output') or '')
    if results:
        tail = 'Commands run:\n' + '\n'.join(results)
    elif assistant.get('commands'):
        tail = 'Commands submitted:\n' + assistant['commands']
    else:
        tail = ''
    return '\n\n'.join(part for part in (explanation, tail) if part)


def _turns(history):
    """The history cut into turns: each begins at a user message."""
    turns = []
    for message in history:
        if message.get('role') == 'user' or not turns:
            turns.append([])
        turns[-1].append(message)
    return turns


def _turn_messages(turn, *, whole):
    """A turn as the messages a request holds: whole, or without its images and command output."""
    messages = []
    for message in turn:
        if whole:
            messages.append({'role': message['role'], 'content': message_content(message['content'], message.get('attachments'))})
        elif message['role'] == 'user':
            text = message['content']
            images = len(message.get('attachments') or [])
            if images:
                text += f"\n[{images} image(s) attached here are not sent again]"
            messages.append({'role': 'user', 'content': text})
        else:
            messages.append({'role': message['role'], 'content': _summarized(message)})
    return messages


def _tokens(messages):
    total = 0
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            total += estimate_tokens(content)
        else:
            total += sum(estimate_tokens(part['text']) if part['type'] == 'text' else IMAGE_TOKENS for part in content)
    return total


def history_messages(history, max_turns=HISTORY_MAX_TURNS, budget_tokens=HISTORY_TOKEN_BUDGET):
    """The part of the conversation a request re-sends: at most max_turns turns within budget_tokens.

    The newest turn is sent whole, with its images and command output; if that alone is over the budget
    its command output is summarized as an older turn's is. Older turns carry only the user's text, the
    explanation and a short summary of their commands' results, and are added, newest first, while
    they fit. The history itself is not changed.
    """
    turns = _turns(history)
    if not turns or max_turns < 1:
        return []
    newest = _turn_messages(turns[-1], whole=True)
    if _tokens(newest) > budget_tokens:
        newest = [({'role': message['role'], 'content': _summarized(source)} if message['role'] != 'user' else message)
                  for message, source in zip(newest, turns[-1])]
    spent = _tokens(newest)
    older = []
    for turn in reversed(turns[:-1]):
        if len(older) + 1 >= max_turns:
            break
        messages = _turn_messages(turn, whole=False)
        spent += _tokens(messages)
        if spent > budget_tokens:
            break
        older.append(messages)
    return [message for turn in reversed(older) for message in turn] + newest
