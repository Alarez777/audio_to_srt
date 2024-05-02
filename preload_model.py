import os
import whisperx
import json

model = whisperx.load_model("large-v2", "cpu", compute_type="float32")
