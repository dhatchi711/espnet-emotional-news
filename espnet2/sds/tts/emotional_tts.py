import re
import sys
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from gdown import download
from transformers import AutoTokenizer
from typeguard import typechecked
from yacs import config as CONFIG

from espnet2.sds.tts.abs_tts import AbsTTS
from espnet2.sds.tts.prompt_tts.prompt_tts_modified.jets import JETSGenerator
from espnet2.sds.tts.prompt_tts.prompt_tts_modified.simbert import StyleEncoder
from espnet2.sds.tts.prompt_tts.prompt_tts_modified.text_2_phoneme import Text2Phoneme

MAX_WAV_VALUE = 32768.0
FS = 16000
EMOTION_LABELS = ["Neutral", "Happy", "Sad", "Angry", "Surprise"]

PRETRAINED_MODEL_GOOGLE_DRIVE = "https://drive.google.com/uc?id=1pWgj6_sOBpvF6EjPt89qGf_kdZq4rVWY"
RESOURCE_DIR = Path(__file__).parents[3] / "resources"

class EmotionalTTS(AbsTTS):
    @typechecked
    def __init__(
        self,
        speaker: str ="0011", # choose 0011, 0012, ..., 0020
        device: str ="cuda",
    ):
        if not RESOURCE_DIR.exists():
            RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
            
        model_dir = RESOURCE_DIR / "emotional_tts"
        if not model_dir.exists():
            download(PRETRAINED_MODEL_GOOGLE_DRIVE, "emotional_tts.zip")
            print("Extracting zip file...", end=" ")
            with zipfile.ZipFile("emotional_tts.zip") as zf:
                zf.extractall(RESOURCE_DIR)
            print("DONE!")

        try:
            sys.path.append(
                str(model_dir / "config")
            )
            from config import Config  # type: ignore
        except ModuleNotFoundError as e:
            print("Error: resources/emotional_tts is not installed.")
            raise e
        super().__init__()
        
        # load config
        config = Config()  # noqa: F841

        with open(config.model_config_path, 'r') as fin: # config.yaml を取得
            conf = CONFIG.load_cfg(fin)

        conf.n_vocab = config.n_symbols
        conf.n_speaker = config.speaker_n_labels

        # load style encoder
        self.style_encoder = StyleEncoder(config).to(device)
        model_CKPT = torch.load(config.style_encoder_ckpt, map_location=device) # style encoder の学習済みモデルを取得
        model_ckpt = {}
        for key, value in model_CKPT['model'].items():
            new_key = key[7:]
            model_ckpt[new_key] = value
        self.style_encoder.load_state_dict(model_ckpt, strict=False)

        # load generator
        generator_checkpoint_path = config.ROOT_DIR / "g_00020000"
        self.text2speech = JETSGenerator(conf).to(device)

        model_CKPT = torch.load(generator_checkpoint_path, map_location=device)
        self.text2speech.load_state_dict(model_CKPT['generator'])
        self.text2speech.eval()

        with open(config.token_list_path, 'r') as f:
            self.token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

        with open(config.speaker2id_path, encoding='utf-8') as f:
            speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.t2p = Text2Phoneme(config)

        self.speaker = speaker2id[speaker]
        self.transcript_regex = re.compile(r"<(.*?)>(.*)")
        self.fs = FS
        self.device = device

    def warmup(self):
        prompt = "Neutral"
        content = "This is warmup."
        text = self.t2p(content)

        style_embedding = self.get_style_embedding(prompt)
        content_embedding = self.get_style_embedding(content)
        text_int = [self.token2id[phoneme] for phoneme in text]

        sequence = torch.from_numpy(np.array(text_int)).to(self.device).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)])).to(self.device)
        style_embedding = torch.from_numpy(style_embedding).to(self.device).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).to(self.device).unsqueeze(0)
        speaker = torch.from_numpy(np.array([self.speaker])).to(self.device)

        with torch.no_grad():
            _ = self.text2speech(
                inputs_ling=sequence,
                inputs_style_embedding=style_embedding,
                input_lengths=sequence_len,
                inputs_content_embedding=content_embedding,
                inputs_speaker=speaker,
                alpha=1.0
            )

    def get_style_embedding(self, prompt: str):
        prompt = self.tokenizer([prompt], return_tensors="pt")
        input_ids = prompt["input_ids"]
        token_type_ids = prompt["token_type_ids"]
        attention_mask = prompt["attention_mask"]

        with torch.no_grad():
            output = self.style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        style_embedding = output["pooled_output"].cpu().squeeze().numpy()
        return style_embedding

    def split_emotion_label_transcript(self, transcript: str) -> Tuple[str, str]:
        regex_result = self.transcript_regex.match(transcript)

        if regex_result is None:
            print(
                "Transcript must follow the format: \"<EMOTION_LABEL>I live in Pittsburgh. ...\", where emotion label must be chosen from [\"Neutral\", \"Happy\", \"Sad\", \"Angry\", \"Surprise\"]."
            )
            raise RuntimeError    

        emotion_label = regex_result.group(1)
        content = regex_result.group(2)

        return emotion_label, content

    def forward(self, transcript: str) -> Tuple[int, np.ndarray]:
        """
        Args:
            transcript (str):
                An input transcript must follow the format:
                    "<EMOTION_LABEL>I live in Pittsburgh. ...",
                where emotion label must be chosen from ["Neutral", "Happy", "Sad", "Angry", "Surprise"].
        
        Returns:
            Tuple[int, np.ndarray]:
                A tuple containing:
                - The sample rate of the audio (int).
                - The generated audio waveform as a
            NumPy array of type `int16`.
        """
        emotion_label, content = self.split_emotion_label_transcript(transcript)
        text = self.t2p(content)

        style_embedding = self.get_style_embedding(emotion_label)
        content_embedding = self.get_style_embedding(content)
        text_int = [self.token2id[phoneme] for phoneme in text]

        sequence = torch.from_numpy(np.array(text_int)).to(self.device).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)])).to(self.device)
        style_embedding = torch.from_numpy(style_embedding).to(self.device).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).to(self.device).unsqueeze(0)
        speaker = torch.from_numpy(np.array([self.speaker])).to(self.device)

        with torch.no_grad():
            infer_output = self.text2speech(
                inputs_ling=sequence,
                inputs_style_embedding=style_embedding,
                input_lengths=sequence_len,
                inputs_content_embedding=content_embedding,
                inputs_speaker=speaker,
                alpha=1.0
            )

            audio_chunk = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
            audio_chunk = audio_chunk.cpu().numpy().astype('int16')

            return (self.fs, audio_chunk)