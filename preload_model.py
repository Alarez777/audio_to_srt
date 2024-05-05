import whisperx

#model = whisperx.load_model("large-v2", "cpu", compute_type="float32")
model = whisperx.load_model("large-v2", "cpu", compute_type="float32", language="es")
audio = whisperx.load_audio("./test.aac")
transcription = model.transcribe(audio, batch_size=5)
model_a, metadata = whisperx.load_align_model(
            language_code=self.transcription["language"], device=self.device)
transcription = whisperx.align(
            transcription["segments"],
            model_a,
            metadata,
            audio,
            cpu,
            return_char_alignments=False,
        )