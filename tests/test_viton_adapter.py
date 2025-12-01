from app.viton_adapter import VITONAdapter
from PIL import Image


def test_viton_adapter_fallback_overlay():
    adapter = VITONAdapter(viton_root="/nonexistent/viton")

    # create a person and cloth small images
    person = Image.new("RGB", (128, 192), color=(255, 230, 200))
    cloth = Image.new("RGB", (64, 64), color=(0, 128, 255))

    out = adapter.run(person, cloth)
    assert out is not None
    assert isinstance(out, Image.Image)
    # Output size should equal person size
    assert out.size == person.size
