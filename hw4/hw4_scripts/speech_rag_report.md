# Speech and RAG workflow

## End-to-end flow

The speech input and output programs use the shared `rag.rag.answer_question`
function, which retrieves syllabus text from Chroma and sends the question and
retrieved context to `gpt-oss` on Nautilus.

```text
spoken question -> STT transcript -> RAG retrieval and answer
typed question  -----------------> RAG retrieval and answer -> TTS WAV -> playback
```

`stt_rag.py` uses local Whisper for speech input. `stt_rag_2.py` follows the
Google Speech Recognition example in `NLP_agent_examples/NLP_STT_sr.py`.
`tts_rag.py` follows `NLP_agent_examples/NLP_TTS.py`: it uses `pyttsx3` to
synthesize the RAG answer, saves a WAV file, then attempts playback with
`sounddevice`.

## Run and capture evidence

Use the syllabus PDF and an audio recording that are accessible from the
Hypercluster. Save the terminal output so the transcript, question, answer,
and playback status can be reviewed. A terminal screenshot can be included in
place of, or alongside, each text log.

Speech input with the Whisper example:

```bash
.venv/bin/python hw4/hw4_scripts/stt_rag.py \
  hw4/hw4_scripts/rag/syllabus.pdf --audio question.wav \
  2>&1 | tee stt_run.txt
```

Speech input using the Google Speech Recognition example:

```bash
.venv/bin/python hw4/hw4_scripts/stt_rag_2.py \
  hw4/hw4_scripts/rag/syllabus.pdf --audio question.wav \
  2>&1 | tee stt_run_google.txt
```

Text question to spoken RAG answer:

```bash
.venv/bin/python hw4/hw4_scripts/tts_rag.py \
  hw4/hw4_scripts/rag/syllabus.pdf \
  --question "What topics does the course cover?" \
  --audio-out tts_answer.wav \
  2>&1 | tee tts_run.txt
```

The STT log should show `Transcription:` and `Answer:`. The TTS log should
show `Question:`, `Answer:`, the generated audio path, and either
`Playback: completed` or an explanation that playback was unavailable. Keep
`question.wav`, the relevant log files or screenshots, and `tts_answer.wav`
with the report as the run evidence.

## Audio device limitation

The audio-file workflow avoids needing microphone access on the Hypercluster.
If playback is also unavailable, document the actual playback message from
`tts_run.txt` or the terminal capture. Include `tts_answer.wav` as evidence
that speech was generated, and play that file on a device with speakers if
possible. Do not report playback as successful unless the run printed
`Playback: completed` and the audio was heard.

## Run results to fill in

Before submitting, replace this section with the actual question, the
transcription (for the STT run), the RAG answer, the playback result, and the
names of the screenshots or log/audio files included as evidence. This keeps
the report tied to the real run rather than sample output.
