"""Demo + sanity test for markdown stripping on the TTS path.

Runs a fixed set of markdown-laced inputs through `strip_markdown` and the
real `stream_pcm` synthesis call, prints before/after, and writes one WAV per
case under `out/tts_markdown_test/` so the audio output can be listened to.

Run from the repo root:
    python -m backend.test_tts_markdown
"""

import os
import sys
import wave

from strip_markdown import strip_markdown

from tts_service import get_sample_rate, stream_pcm


CASES: list[tuple[str, str]] = [
    ("plain_control", "This is a plain sentence with no markdown at all."),
    ("dash_bullets", "- first item\n- second item\n- third item"),
    ("star_bullets", "* alpha\n* beta\n* gamma"),
    ("plus_bullets", "+ red\n+ green\n+ blue"),
    ("numbered_list", "1. Preheat the oven.\n2. Mix the batter.\n3. Bake for thirty minutes."),
    ("atx_headers", "# Title\n## Subtitle\n### Section three\nbody text follows."),
    ("bold_star", "This is **very important** information."),
    ("bold_underscore", "This is __also important__ information."),
    ("italic_star", "Use the *gentle* setting."),
    ("italic_underscore", "Use the _quiet_ setting."),
    ("inline_code", "Run the `npm install` command first."),
    ("fenced_code", "Here is some code:\n```python\nprint('hi')\n```\nThat is the example."),
    ("link", "See the [official docs](https://example.com/docs) for details."),
    ("image", "Logo: ![company logo](https://example.com/logo.png) end."),
    ("blockquote", "> Wisdom is knowing what to leave out.\nThat is the quote."),
    ("horizontal_rule", "Section one.\n\n---\n\nSection two."),
    ("mixed_paragraph",
     "## Tips for baking\n"
     "- Use **room-temperature** eggs.\n"
     "- Sift the *dry* ingredients.\n"
     "- Don't open the `oven` door early.\n"
     "See [this guide](https://example.com) for more."),
    ("empty_after_strip", "**__``__**"),
]


def _synthesize_to_wav(text: str, path: str, sample_rate: int) -> int:
    """Run text through stream_pcm and write the int16 PCM bytes to a WAV.
    Returns the total number of audio bytes written (0 if nothing was synthesized)."""
    audio = b"".join(stream_pcm(text))
    if not audio:
        return 0
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    return len(audio)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.path.join(repo_root, "out", "tts_markdown_test")
    os.makedirs(out_dir, exist_ok=True)

    sr = get_sample_rate()
    print(f"Piper sample rate: {sr} Hz")
    print(f"Output directory:  {out_dir}")
    print("=" * 72)

    summary: list[tuple[str, int, str]] = []
    for idx, (label, raw) in enumerate(CASES, start=1):
        stripped = strip_markdown(raw).strip()
        print(f"\n[{idx:02d}] {label}")
        print(f"  INPUT:    {raw!r}")
        print(f"  STRIPPED: {stripped!r}")

        wav_path = os.path.join(out_dir, f"{idx:02d}_{label}.wav")
        # Pass the RAW markdown to stream_pcm to exercise the in-pipeline
        # strip (the final safety net in tts_service.stream_pcm).
        n_bytes = _synthesize_to_wav(raw, wav_path, sr)
        if n_bytes:
            print(f"  WAV:      {wav_path}  ({n_bytes} bytes PCM)")
        else:
            print(f"  WAV:      <skipped — nothing to synthesize>")
            wav_path = ""
        summary.append((label, n_bytes, wav_path))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'#':>2}  {'case':<22}  {'bytes':>9}  path")
    for i, (label, n_bytes, wav_path) in enumerate(summary, start=1):
        print(f"{i:>2}  {label:<22}  {n_bytes:>9}  {wav_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
