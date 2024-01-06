import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from google.colab import drive
from IPython.display import Audio
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import ModelOutput
from transformers import  Wav2Vec2FeatureExtractor, AutoConfig
from transformers.models.hubert.modeling_hubert import (
    HubertPreTrainedModel,
    HubertModel
)
from typing import Tuple, List, Any

# The legacy of the model's author
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


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


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()

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
        outputs = self.hubert(
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


def load_hubert_model() -> Tuple[HubertForSpeechClassification, Wav2Vec2FeatureExtractor, AutoConfig, torch.device]:
    """
    Loads the HuBERT model and related components.

    Returns:
        Tuple containing the model, feature extractor, configuration, and device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion").to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    config = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")
    return model, feature_extractor, config, device


def classify_emotion(sound_array: np.ndarray, sampling_rate: int, model: HubertForSpeechClassification,
                     feature_extractor: Wav2Vec2FeatureExtractor, config: AutoConfig, device: torch.device) -> int:
    """
    Classifies the emotion based on audio data.

    Args:
        sound_array: Audio data for prediction.
        sampling_rate: Sampling rate of the audio data.
        model: The loaded HuBERT model.
        feature_extractor: Feature extractor for the model.
        config: Model configuration.
        device: The device on which the model is executed.

    Returns:
        Predicted emotion label as an int.
    """
    inputs = feature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0]
    predicted_label = int(np.argmax(scores))
    return predicted_label


def run_hubert_inference(df: pd.DataFrame, model: HubertForSpeechClassification,
                         feature_extractor: Wav2Vec2FeatureExtractor, config: AutoConfig, device: torch.device) -> List[
    int]:
    """
    Runs HuBERT model inference on a DataFrame containing audio data.

    Args:
        df: DataFrame containing audio data and labels.
        model: The loaded HuBERT model.
        feature_extractor: The feature extractor.
        config: Model configuration.
        device: The device for running the model.

    Returns:
        List of predicted labels as strings.
    """
    return [classify_emotion(row['audio'], 16000, model, feature_extractor, config, device) for _, row in df.iterrows()]


# if __name__ == "__main__":
#     df = ...  # your DataFrame, ensure it has an 'audio' column
#     model, feature_extractor, config, device = load_hubert_model()
#     predicted_labels = run_hubert_inference(df, model, feature_extractor, config, device)
#     for true_label, pred_label in zip(df['label'], predicted_labels):  # Assuming 'label' column in df
#         print(f"True: {true_label}, Predicted: {pred_label}")
