"""Ask the syllabus RAG app a question, synthesize its answer, and play it.

Text-to-speech uses ``pyttsx3`` following the example in
``NLP_agent_examples/NLP_TTS.py``. The spoken answer is also saved as a WAV
file so the run can be reviewed or used when the environment has no speaker.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

from rag.rag import DEFAULT_PERSIST_DIRECTORY, answer_question


def synthesize_answer(text, output_path="tts_answer.wav", rate=150, voice_index=None):
    """Synthesize text to a WAV file using pyttsx3 and return the output path."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("answer text must be a non-empty string")

    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError(
            "pyttsx3 is required for text-to-speech. Install it with "
            "`pip install pyttsx3`."
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        engine = pyttsx3.init()
    except Exception as exc:
        raise RuntimeError(
            "Could not start the pyttsx3 voice engine. On Linux, install an "
            "eSpeak voice engine (for example, espeak-ng)."
        ) from exc

    try:
        engine.setProperty("rate", rate)
        if voice_index is not None:
            voices = engine.getProperty("voices")
            if voice_index < 0 or voice_index >= len(voices):
                raise ValueError(
                    f"voice index must be between 0 and {len(voices) - 1}"
                )
            engine.setProperty("voice", voices[voice_index].id)
        engine.save_to_file(text.strip(), str(output))
        engine.runAndWait()
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not synthesize the answer: {exc}") from exc
    finally:
        engine.stop()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"TTS did not produce an audio file at {output}")
    return output


def play_wav(audio_path):
    """Play a PCM WAV using sounddevice; return (success, reason)."""
    try:
        import numpy as np
        import sounddevice as sd

        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        dtype_by_width = {1: np.uint8, 2: np.int16, 4: np.int32}
        if sample_width not in dtype_by_width:
            return False, f"unsupported WAV sample width ({sample_width} bytes)"

        audio = np.frombuffer(frames, dtype=dtype_by_width[sample_width])
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, sample_rate)
        sd.wait()
        return True, "played"
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(
        description="Ask the syllabus RAG app a question and speak its answer."
    )
    parser.add_argument("syllabus_pdf", help="path to the syllabus PDF")
    parser.add_argument(
        "--question", help="question to send to RAG; prompts interactively if omitted"
    )
    parser.add_argument(
        "--audio-out",
        default="tts_answer.wav",
        help="where to save the synthesized answer (default: tts_answer.wav)",
    )
    parser.add_argument(
        "--rate", type=int, default=150, help="speech rate in words per minute"
    )
    parser.add_argument(
        "--voice-index", type=int, help="optional pyttsx3 voice index to select"
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="save the spoken answer without playing it",
    )
    parser.add_argument(
        "--persist-directory",
        default=str(DEFAULT_PERSIST_DIRECTORY),
        help="directory for the persistent Chroma index",
    )
    args = parser.parse_args()

    question = args.question if args.question is not None else input("Question: ")
    if not question.strip():
        parser.error("the question must not be empty")
    if args.rate <= 0:
        parser.error("--rate must be greater than zero")

    try:
        print(f"Question: {question.strip()}")
        answer = answer_question(
            question,
            args.syllabus_pdf,
            persist_directory=args.persist_directory,
        )
        if not answer or not answer.strip():
            raise RuntimeError("RAG returned an empty answer; there is nothing to speak.")
        print(f"Answer: {answer}")

        audio_path = synthesize_answer(
            answer,
            output_path=args.audio_out,
            rate=args.rate,
            voice_index=args.voice_index,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"TTS/RAG error: {exc}", file=sys.stderr)
        return 2

    print(f"Audio file: {audio_path}")
    if args.no_playback:
        print("Playback: skipped (--no-playback)")
    else:
        played, reason = play_wav(audio_path)
        if played:
            print("Playback: completed")
        else:
            print(f"Playback unavailable: {reason}")
            print(f"Use the saved WAV file as audio-file evidence: {audio_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
