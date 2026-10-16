"""Stage 4's frames: the poster pipeline, copied, under the posters' library versions (README trap 3)."""

import io
import re

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from sharerate import data, frames

# The IJG base quantisation tables (JPEG standard, Annex K), row by row, the order Pillow reports them in.
_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
         29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
         121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_CHROMA = [17, 18, 24, 47] + [99] * 4 + [18, 21, 26, 66] + [99] * 4 + [24, 26, 56] + [99] * 5 + [47, 66] + [99] * 38


def _ijg_table(base, quality):
    """libjpeg's quality scaling: what 'JPEG quality 82' means, independent of Pillow."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return [min(max((b * scale + 50) // 100, 1), 255) for b in base]


def _video(path, seconds, fps=10, gop=5):
    """A small test clip whose frame i is a flat grey of level 8 * i, with a keyframe every `gop` frames."""
    import av

    with av.open(str(path), "w") as container:
        # sc_threshold off: without it the encoder adds a keyframe at every change of grey, so at every frame
        stream = container.add_stream("mpeg4", rate=fps, options={"sc_threshold": "1000000000"})
        stream.width, stream.height, stream.pix_fmt = 320, 576, "yuv420p"
        stream.codec_context.gop_size = gop
        for i in range(int(seconds * fps)):
            pixels = np.full((576, 320, 3), 8 * i, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _grey(jpeg):
    return float(np.asarray(Image.open(io.BytesIO(jpeg)).convert("L")).mean())


def test_the_environment_has_the_posters_library_versions():
    frames.check_versions()  # raises if av or Pillow is not the version the posters were made with


def test_a_frame_is_480_px_on_its_longest_edge_at_jpeg_quality_82():
    jpeg = frames.encode_poster(Image.new("RGB", (1080, 1920), (120, 30, 200)))
    image = Image.open(io.BytesIO(jpeg))
    assert image.format == "JPEG" and image.size == (270, 480)  # a 9:16 reel, its height cut to 480
    assert image.quantization[0] == _ijg_table(_LUMA, 82)
    assert image.quantization[1] == _ijg_table(_CHROMA, 82)
    assert JpegImagePlugin.get_sampling(image) == 2  # 4:2:0 chroma, Pillow's default at this quality


def test_a_small_image_is_never_enlarged():
    image = Image.open(io.BytesIO(frames.encode_poster(Image.new("RGB", (200, 300)))))
    assert image.size == (200, 300)


def test_the_frame_is_the_keyframe_at_or_before_one_second(tmp_path):
    clip = tmp_path / "clip.mp4"
    _video(clip, seconds=3, gop=5)  # keyframes at 0.0, 0.5, 1.0, 1.5 s
    assert abs(_grey(frames.frame_from_video(clip)) - 80) < 4  # frame 10, at 1.0 s


def test_without_a_keyframe_near_one_second_the_frame_is_the_first(tmp_path):
    clip = tmp_path / "clip.mp4"
    _video(clip, seconds=3, gop=100)  # one keyframe, at 0 s: the seek lands there
    assert _grey(frames.frame_from_video(clip)) < 4


def test_a_clip_of_two_seconds_or_less_is_not_seeked(tmp_path):
    clip = tmp_path / "clip.mp4"
    _video(clip, seconds=1.5, gop=5)
    assert _grey(frames.frame_from_video(clip)) < 4


@pytest.mark.skipif(not (frames.FLYWHEEL_DIR / "Backend").exists(), reason="the flywheel is not on this disk")
def test_the_settings_still_match_the_flywheels_poster_code():
    # Read-only. If the platform ever changes how it makes posters, the frames made here must change with it.
    source = (frames.FLYWHEEL_DIR / "Backend/app/domains/platforms/instagram/competitors/thumbnail_service.py"
              ).read_text(encoding="utf-8")
    assert re.search(r"^_THUMBNAIL_MAX_EDGE = (\d+)", source, re.M).group(1) == str(frames.THUMBNAIL_MAX_EDGE)
    assert re.search(r"^_JPEG_QUALITY = (\d+)", source, re.M).group(1) == str(frames.JPEG_QUALITY)
    assert float(re.search(r"^_POSTER_SEEK_SECONDS = ([\d.]+)", source, re.M).group(1)) == frames.POSTER_SEEK_SECONDS
    assert 'image.save(tmp_path, format="JPEG", quality=_JPEG_QUALITY, optimize=True)' in source
    assert "Image.Resampling.LANCZOS" in source


@pytest.mark.skipif(not frames.FRAME_MAP.exists(), reason="the frames are not made on this disk")
def test_frame_paths_name_the_frame_and_never_its_source():
    paths = frames.frame_paths()
    assert len(paths) == len(data.load_videos())
    stems = paths.dropna().map(lambda p: p.replace("\\", "/").rsplit("/", 1)[1])
    assert stems.str.fullmatch(r"[0-9a-f]{32}\.jpg").all()  # a random id: no username, no video id
    assert paths.name == "frame_path"
