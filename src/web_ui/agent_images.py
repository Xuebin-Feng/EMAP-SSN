# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Image validation shared by the upload endpoint and agent messages."""
import base64
import io
import warnings
import xml.etree.ElementTree as ET

from PIL import Image

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ATTACHMENTS = 10
MAX_DIMENSION = 1600


def inspect_source(raw):
    """Reject animation before the browser rasterizes the original image."""
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('Each image must be nonempty and at most 20 MiB.')
    # SVG is decoded by the browser, not Pillow. Only self-contained static SVG
    # is accepted: scripts and embedded resources could animate indirectly.
    if raw.lstrip().startswith((b'<', b'\xef\xbb\xbf', b'\xff\xfe', b'\xfe\xff')):
        try:
            root = ET.fromstring(raw)
            if root.tag.split('}')[-1] != 'svg':
                raise ValueError('Only image files are accepted.')
            for element in root.iter():
                tag = element.tag.split('}')[-1].lower()
                if tag in {'animate', 'animatemotion', 'animatetransform', 'set',
                           'script', 'foreignobject', 'image'}:
                    raise ValueError('SVG must be static and self-contained, without animation or embedded images.')
                if tag == 'style' and any(token in (element.text or '').lower()
                                          for token in ('animation', 'transition', '@', '\\', 'url(')):
                    raise ValueError('Animated or external SVG styles are not accepted.')
                for key, value in element.attrib.items():
                    key = key.split('}')[-1].lower()
                    if (key.startswith('on') or any(token in value.lower() for token in ('animation', 'transition', '\\'))
                            or (key == 'href' and not value.startswith('#'))):
                        raise ValueError('SVG must be static and self-contained.')
            return {'mime_type': 'image/svg+xml'}
        except ET.ParseError as error:
            raise ValueError('Invalid or unsupported image file.') from error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as img:
                if getattr(img, 'n_frames', 1) != 1 or getattr(img, 'is_animated', False):
                    raise ValueError('Animated and multi-frame images are not accepted.')
                if img.format == 'ICO' and len(img.ico.sizes()) > 1:
                    raise ValueError('Multi-frame icons are not accepted.')
                mime = Image.MIME.get(img.format, 'application/octet-stream')
                img.load()
                return {'mime_type': mime}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('Invalid or unsupported image file; only decodable static images are accepted.') from error


def validate_attachments(attachments):
    if attachments is None:
        return []
    if not isinstance(attachments, list) or len(attachments) > MAX_ATTACHMENTS:
        raise ValueError('A message can contain at most 10 image attachments.')
    result = []
    prefix = 'data:image/png;base64,'
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError('Invalid image attachment.')
        url = attachment.get('data_url', '')
        if not isinstance(url, str) or not url.startswith(prefix) or len(url) > MAX_IMAGE_BYTES * 4 // 3 + 100:
            raise ValueError('Attachments must be PNG images of at most 20 MiB.')
        try:
            raw = base64.b64decode(url[len(prefix):], validate=True)
            info = inspect_source(raw)
            with Image.open(io.BytesIO(raw)) as img:
                if info['mime_type'] != 'image/png' or max(img.size) > MAX_DIMENSION:
                    raise ValueError('Attachments must be PNG images no larger than 1600 pixels per side.')
        except ValueError:
            raise
        except Exception as error:
            raise ValueError('Invalid PNG attachment.') from error
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


def history_messages(history):
    return [{'role': msg['role'], 'content': message_content(msg['content'], msg.get('attachments'))}
            for msg in history]
