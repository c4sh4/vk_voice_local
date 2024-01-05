import torch
import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSpeechClassification, AutoConfig
from tqdm import tqdm
from torch.nn import functional as F


# Вот эта штука капец нужна
# !git clone https://github.com/m3hrdadfi/soxan.git
# %cd soxan
# !pip install -r requirements.txt
# from soxan.src.models import Wav2Vec2ForSpeechClassification

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
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        audio_data = row['audio']
        predicted_label, scores, _ = predict_emotion(model, feature_extractor, config, device, audio_data)
        df.at[index, 'predicted_label'] = predicted_label
        df.at[index, 'predicted_scores'] = scores
    return df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_extractor, config, sampling_rate = load_model(device)

    df_for_soxan = ...  # Load your DataFrame here
    df_for_soxan['predicted_label'] = None
    df_for_soxan['predicted_scores'] = None

    df_for_soxan = run_inference_on_dataframe(df_for_soxan, model, feature_extractor, config, device)
    print(df_for_soxan)
