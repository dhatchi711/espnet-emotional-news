import numpy as np
import torch.nn as nn

from espnet2.sds.tts.prompt_tts.hifigan.get_random_segments import get_random_segments
from espnet2.sds.tts.prompt_tts.hifigan.models import Generator as HiFiGANGenerator
from espnet2.sds.tts.prompt_tts.prompt_tts_modified.model_open_source import PromptTTS


class JETSGenerator(nn.Module):
    def __init__(self, config) -> None:

        super().__init__()
        
        self.upsample_factor=int(np.prod(config.model.upsample_rates))

        self.segment_size = config.segment_size

        self.am = PromptTTS(config)

        self.generator = HiFiGANGenerator(config.model)

        self.config=config


    def forward(
        self, 
        inputs_ling, 
        input_lengths, 
        inputs_speaker, 
        inputs_style_embedding, 
        inputs_content_embedding, 
        mel_targets=None, 
        output_lengths=None, 
        pitch_targets=None, 
        energy_targets=None, 
        alpha=1.0, 
        cut_flag=True
    ):
        outputs = self.am(inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding , inputs_content_embedding, mel_targets , output_lengths , pitch_targets , energy_targets , alpha)


        if mel_targets is not None and cut_flag:
            z_segments, z_start_idxs, segment_size = get_random_segments(
                outputs["dec_outputs"].transpose(1,2),
                output_lengths,
                self.segment_size,
            )
        else:
            z_segments = outputs["dec_outputs"].transpose(1,2)
            z_start_idxs=None
            segment_size=self.segment_size

        wav = self.generator(z_segments)

        outputs["wav_predictions"] = wav
        outputs["z_start_idxs"]= z_start_idxs
        outputs["segment_size"] = segment_size
        return outputs
