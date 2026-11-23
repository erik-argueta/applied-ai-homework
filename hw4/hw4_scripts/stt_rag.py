"""Transcribe a spoken syllabus question and pass it to the RAG application.

Provide either an audio file or use ``--record`` to capture audio from a
microphone. Transcription uses the local OpenAI Whisper package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.rag import DEFAULT_PERSIST_DIRECTORY, answer_question


def transcribe_audio(audio_path=None, duration=5, model_name="base"):
    """Transcribe an audio file or microphone recording with Whisper."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is required for speech transcription. Install it with "
            "`pip install openai-whisper`."
        ) from exc

    model = whisper.load_model(model_name)
    if audio_path is not None:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        result = model.transcribe(str(path), fp16=False, language="en")
    else:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Microphone recording requires sounddevice. Install it with "
                "`pip install sounddevice`."
            ) from exc

        sample_rate = 16_000
        print(f"Recording for {duration:g} seconds. Speak your syllabus question.")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        result = model.transcribe(audio.flatten(), fp16=False, language="en")

    transcript = result.get("text", "").strip()
    if not transcript:
        raise RuntimeError("Whisper did not detect any speech in the audio.")
    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a spoken question and answer it with the syllabus RAG app."
    )
    parser.add_argument("syllabus_pdf", help="path to the syllabus PDF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="path to a spoken question audio file")
    source.add_argument(
        "--record",
        action="store_true",
        help="record a spoken question from the default microphone",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5,
        help="recording length in seconds when using --record (default: 5)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size to load (default: base)",
    )
    parser.add_argument(
        "--persist-directory",
        default=str(DEFAULT_PERSIST_DIRECTORY),
        help="directory for the persistent Chroma index (default: ./chroma_test)",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    try:
        transcript = transcribe_audio(
            audio_path=args.audio,
            duration=args.duration,
            model_name=args.whisper_model,
        )
        print(f"Transcription: {transcript}")
        answer = answer_question(
            transcript,
            args.syllabus_pdf,
            persist_directory=args.persist_directory,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"STT/RAG error: {exc}", file=sys.stderr)
        return 2

    print(f"Answer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
