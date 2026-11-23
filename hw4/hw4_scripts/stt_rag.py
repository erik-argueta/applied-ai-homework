"""Send a spoken syllabus question through STT and the RAG application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_solo import answer_question


def transcribe_audio(
    audio_path: str | None, duration: float, language: str
) -> str:
    """Transcribe an audio file or microphone recording with SpeechRecognition."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise RuntimeError(
            "SpeechRecognition is required. Install it with "
            "`pip install SpeechRecognition`."
        ) from exc

    recognizer = sr.Recognizer()
    if audio_path:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            with sr.AudioFile(str(path)) as source:
                audio_data = recognizer.record(source)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Could not read the audio file. Use a WAV, AIFF, or FLAC file."
            ) from exc
    else:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Microphone recording requires sounddevice. Install it with "
                "`pip install sounddevice`."
            ) from exc

        sample_rate = 16_000
        print(f"Recording for {duration:g} seconds. Speak your question.")
        try:
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            raise RuntimeError(f"Could not record from the microphone: {exc}") from exc

        pcm = (np.clip(audio.flatten(), -1.0, 1.0) * 32767).astype(np.int16)
        audio_data = sr.AudioData(pcm.tobytes(), sample_rate, 2)

    try:
        transcript = recognizer.recognize_google(audio_data, language=language).strip()
    except sr.UnknownValueError as exc:
        raise RuntimeError("Speech Recognition could not understand the audio.") from exc
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech Recognition service request failed: {exc}") from exc

    if not transcript:
        raise RuntimeError("Speech Recognition returned an empty transcript.")
    return transcript


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a spoken question, send it to the syllabus RAG app, "
            "and display the answer."
        )
    )
    audio_source = parser.add_mutually_exclusive_group(required=True)
    audio_source.add_argument(
        "--audio", help="path to a WAV, AIFF, or FLAC question recording"
    )
    audio_source.add_argument(
        "--record",
        action="store_true",
        help="record a question from the default microphone",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5,
        help="recording length in seconds with --record (default: 5)",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="Google Speech Recognition language tag (default: en-US)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="number of relevant syllabus chunks to retrieve (default: 3)",
    )
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    try:
        transcript = transcribe_audio(
            audio_path=args.audio,
            duration=args.duration,
            language=args.language,
        )
        print(f"Transcription: {transcript}")
        answer, _ = answer_question(transcript, top_k=args.top_k)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"STT/RAG error: {exc}", file=sys.stderr)
        return 2

    print(f"Answer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
