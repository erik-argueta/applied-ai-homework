"""Google Speech Recognition example connected to the syllabus RAG app.

This is an alternative to ``stt_rag.py``. It follows the SpeechRecognition
example in ``NLP_agent_examples/NLP_STT_sr.py`` and sends the recognized text
to ``rag.rag.answer_question``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.rag import DEFAULT_PERSIST_DIRECTORY, answer_question


def transcribe_audio(audio_path=None, duration=5, language="en-US"):
    """Transcribe an audio file or sounddevice microphone recording."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise RuntimeError(
            "SpeechRecognition is required. Install it with "
            "`pip install SpeechRecognition`."
        ) from exc

    recognizer = sr.Recognizer()
    if audio_path is not None:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            with sr.AudioFile(str(path)) as source:
                audio_data = recognizer.record(source)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Could not read audio. Use a WAV, AIFF, or FLAC file."
            ) from exc
    else:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Microphone capture requires numpy and sounddevice. Install "
                "sounddevice with `pip install sounddevice`."
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
        pcm = (np.clip(audio.flatten(), -1.0, 1.0) * 32767).astype(np.int16)
        audio_data = sr.AudioData(pcm.tobytes(), sample_rate, 2)

    try:
        return recognizer.recognize_google(audio_data, language=language).strip()
    except sr.UnknownValueError as exc:
        raise RuntimeError("Speech Recognition could not understand the audio.") from exc
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech Recognition service request failed: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a spoken question and answer it with the syllabus RAG app."
    )
    parser.add_argument("syllabus_pdf", help="path to the syllabus PDF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="path to a WAV, AIFF, or FLAC question recording")
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
        "--language",
        default="en-US",
        help="Google Speech Recognition language tag (default: en-US)",
    )
    parser.add_argument(
        "--persist-directory",
        default=str(DEFAULT_PERSIST_DIRECTORY),
        help="directory for the persistent Chroma index",
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
        if not transcript:
            raise RuntimeError("Speech Recognition returned an empty transcript.")

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
