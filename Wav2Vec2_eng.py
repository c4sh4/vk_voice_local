import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2FeatureExtractor, AutoConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from dataclasses import dataclass
from typing import Optional, Tuple


# Legacy


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
# Legacy ended here


def load_model(device: torch.device) -> tuple:
    """
    Loads the model and related components.

    Args:
        device: The device on which the model is executed.

    Returns:
        Tuple containing the model, feature extractor, config, and sampling rate.
    """
    model_name_or_path = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
    config = AutoConfig.from_pretrained(model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    sampling_rate = feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
    return model, feature_extractor, config, sampling_rate


def predict_emotion(model, feature_extractor, config, device, audio_data, sampling_rate=16000):
    """
    Predicts the emotion from the audio data.

    Args:
        model: The loaded model.
        feature_extractor: Feature extractor for the model.
        config: Model configuration.
        device: The device on which the model is executed.
        audio_data: Audio data for prediction.
        sampling_rate: Sampling rate of the audio data.

    Returns:
        Tuple of predicted label, scores, and detailed outputs.
    """
    inputs = feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    predicted_label = np.argmax(scores)
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return predicted_label, scores, outputs


def run_inference_on_dataframe(df: pd.DataFrame, model, feature_extractor, config, device) -> pd.DataFrame:
    """
    Runs the emotion prediction on each row of the DataFrame.

    Args:
        df: DataFrame containing the audio data.
        model: The loaded model.
        feature_extractor: The feature extractor.
        config: Model configuration.
        device: The device for running the model.

    Returns:
        DataFrame with predictions.
    """
    df['predicted'] = None
    df['predicted_scores'] = None
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        audio_data = row['audio']
        predicted_label, scores, _ = predict_emotion(model, feature_extractor, config, device, audio_data)
        df.at[index, 'predicted'] = predicted_label
        df.at[index, 'predicted_scores'] = scores
    return df


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model, feature_extractor, config, sampling_rate = load_model(device)
#     df_for_soxan =
#
#     df_for_soxan = run_inference_on_dataframe(df_for_soxan, model, feature_extractor, config, device)
#     print(df_for_soxan)
