"""Image fixtures shared by the in-Viewer agent tests (test_agent_images.py and
test_agent_composer.py)."""
import io

from PIL import Image


def image_bytes(fmt='PNG', animated=False, size=(20, 10)):
    out = io.BytesIO()
    kwargs = {'save_all': True, 'append_images': [Image.new('RGB', size, 'blue')], 'duration': 100} if animated else {}
    Image.new('RGB', size, 'red').save(out, format=fmt, **kwargs)
    return out.getvalue()
