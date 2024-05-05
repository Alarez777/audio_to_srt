import whisperx

model = whisperx.load_model("large-v2", "cpu", compute_type="float32")
model = whisperx.load_model("large-v2", "cpu", compute_type="float32", language="es")
audio = whisperx.load_audio("./test.aac")