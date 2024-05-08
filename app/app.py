import os
import whisperx
import json
import moviepy.editor as mp
import yt_dlp
import re
import logging

log_level = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=log_level
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

source_name = os.environ.get("SOURCE_NAME", "source.m4a")
language = os.environ.get("LANGUAGE", "unknown")
batch = int(os.environ.get("BATCH", "5"))

path_source = f"/app/source/{source_name}"
logger.info(f"File name: {source_name}")
logger.info(f"Bath size: {batch}")
logger.info(f"Language: {language}")


class Transcriber:
    def __init__(self, device="cpu", batch_size=5, compute_type="float32"):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.file_path = None
        self.transcription = None

    def clear_previous_audio(self):
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)

    def upload_file(self, source_type):
        self.clear_previous_audio()

        if source_type == "social_media":
            url = input("Ingrese la URL de la red social: ")
            self.download_audio_from_url(url)
        elif source_type in {"audio", "video"}:
            # self.file_path = self.upload_file_and_get_path()
            self.file_path = path_source
        else:
            print("Tipo de fuente no valido.")
            return

        if source_type == "video":
            self.extract_audio_from_video()

    # def upload_file_and_get_path(self):
    #     uploaded = widgets.FileUpload(
    #         accept="",
    #         multiple=False,
    #     )
    #     output = widgets.Output()
    #     display(uploaded)
    #     display(output)
    #
    #     def on_upload(change):
    #         with output:
    #             print("Archivo subido:", change.new)
    #
    #     uploaded.observe(on_upload, names="value")
    #
    #     # Esperar a que el archivo sea subido
    #     while uploaded.value is None:
    #         pass
    #     path_file = f"./{uploaded.name}"
    #     with open(path_file, "wb") as fp:
    #         fp.write(uploaded.content)
    #     return path_file

    def download_audio_from_url(self, url):
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": f"extracted_audio.%(ext)s",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        self.file_path = f"extracted_audio.wav"

    def extract_audio_from_video(self):
        clip = mp.VideoFileClip(self.file_path)
        clip.audio.write_audiofile("extracted_audio.wav")
        self.file_path = "extracted_audio.wav"

    def transcribe_audio(self, output_format, language):
        self.output_format = output_format
        if language == "unknown":
            model = whisperx.load_model(
                "large-v2", self.device, compute_type=self.compute_type
            )
        else:
            model = whisperx.load_model(
                "large-v2",
                self.device,
                compute_type=self.compute_type,
                language=language,
            )
        audio = whisperx.load_audio(self.file_path)
        self.transcription = model.transcribe(
            audio, batch_size=self.batch_size, print_progress=True
        )

        model_a, metadata = whisperx.load_align_model(
            language_code=self.transcription["language"], device=self.device
        )
        self.transcription = whisperx.align(
            self.transcription["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        self.save_transcription()

    def save_transcription(self):
        if self.output_format == "txt":
            transcription_text = "\n".join(
                [segment["text"] for segment in self.transcription["segments"]]
            )
            path_output = f"/app/output/{source_name}.txt"
            with open(f"{path_output}", "w", encoding="utf8") as file:
                file.write(transcription_text)

        elif self.output_format == "srt":
            words_per_entry = 3
            srt_content = self.create_srt_content(self.transcription, words_per_entry)
            path_output = f"/app/output/{source_name}.srt"
            with open(f"{path_output}", "w", encoding="utf8") as file:
                file.write(srt_content)

        elif self.output_format == "json":
            path_output = f"/app/output/{source_name}.json"
            with open(f"{path_output}", "w", encoding="utf8") as file:
                json.dump(self.transcription, file, indent=4, ensure_ascii=False)
        else:
            print("Formato no soportado.")

    def create_srt_content(self, data, words_per_entry=5):
        srt_content = ""
        global_idx = 1

        for segment in data["segments"]:
            words = segment["text"].split()
            total_words = len(words)
            segment_duration = segment["end"] - segment["start"]

            entry_words = []
            current_word_count = 0
            is_break_point = False

            for word in words:
                if bool(re.search(r"[,.?]", word)):
                    is_break_point = True
                entry_words.append(word)
                current_word_count += 1

                if current_word_count >= words_per_entry or is_break_point:
                    entry_duration = (
                        current_word_count / total_words
                    ) * segment_duration

                    start_time = self.seconds_to_srt_time(segment["start"])
                    end_time = self.seconds_to_srt_time(
                        segment["start"] + entry_duration
                    )

                    srt_content += f"{global_idx}\n{start_time} --> {end_time}\n{' '.join(entry_words)}\n\n"
                    global_idx += 1

                    segment["start"] += entry_duration
                    entry_words = []
                    current_word_count = 0
                    is_break_point = False

            if entry_words:
                start_time = self.seconds_to_srt_time(segment["start"])
                end_time = self.seconds_to_srt_time(segment["end"])

                srt_content += f"{global_idx}\n{start_time} --> {end_time}\n{' '.join(entry_words)}\n\n"
                global_idx += 1

        return srt_content

    @staticmethod
    def seconds_to_srt_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, remainder = divmod(remainder, 60)
        seconds, milliseconds = divmod(remainder, 1)
        return "{:02}:{:02}:{:02},{:03}".format(
            int(hours), int(minutes), int(seconds), int(milliseconds * 1000)
        )


# @title Transcriptor de Audio { vertical-output: true, display-mode: "form" }
source_type = "audio"  # @param ["audio", "video", "social_media"]
output_format = "srt"  # @param ["txt", "srt", "json"]


transcriber = Transcriber(batch_size=batch)
transcriber.upload_file(source_type=source_type)
transcriber.transcribe_audio(output_format=output_format, language=language)
