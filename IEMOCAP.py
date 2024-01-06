# imports
import os
import librosa
import numpy as np
import pandas as pd


def make_dir(output_dir_path: str) -> str:
    """
    :param output_dir_path:
    :return: output_dir_path:
    """
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    return output_dir_path


def write_session(annotations_dir: str, audio_dir: str, output_dir: str) -> None:
    """
    :param annotations_dir:
    :param audio_dir:
    :param output_dir:
    :return: None
    """
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_dir, audio_file)
            annotation_file = audio_file.replace('.wav', '.txt')
            annotation_path = os.path.join(annotations_dir, annotation_file)
            # check for annotation
            if not os.path.exists(annotation_path):
                print(f"annotation file {annotation_file} doesn't exist.")
                continue
            try:
                # wav load
                y, sr = librosa.load(audio_path, sr=None)
                with open(annotation_path, 'r') as af:
                    for line in af:
                        if line.startswith('['):
                            parts = line.strip().split('\t')
                            times = parts[0].strip('[]').split(' - ')
                            start_time = float(times[0])
                            end_time = float(times[1])
                            label = parts[2]
                            # skip xxx
                            if label == 'xxx':
                                continue
                            start_sample = int(start_time * sr)
                            end_sample = int(end_time * sr)
                            cut_signal = y[start_sample:end_sample]

                            output_filename = f"{audio_file.replace('.wav', '')}_{parts[1]}_{label}.npy"
                            output_filepath = os.path.join(output_dir, output_filename)
                            np.save(output_filepath, cut_signal)

            except Exception as e:
                print(f"Xd error on {audio_file}: {e}")


def create_dataframe_from_npy(directory: str) -> pd.DataFrame:
    data = []
    mapping_iemocap_dict = {
        'neu': 0,
        'fru': 1,
        'ang': 2,
        'sad': 3,
        'hap': 4,
        'exc': 5,
        'sur': 6,
        'oth': 7,
        'fea': 8,
        'dis': 9
    }
    # Перебор всех файлов в директории
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            # Загрузка массива NumPy
            np_array = np.load(os.path.join(directory, file))

            # Извлечение информации из имени файла
            parts = file.split('_')
            session_type = parts[1]  # 'impro' или 'script'
            label_text = parts[-1].replace('.npy', '')  # Метка эмоции
            label_num = mapping_iemocap_dict.get(label_text, -1)
            # Добавление данных в список
            data.append([np_array, session_type, label_text, label_num])
    df = pd.DataFrame(data, columns=['audio', 'session_type', 'labels_text', 'labels'])
    return df


if __name__ == "__main__":
    annotations_dir_ = ''
    # r'C:\Users\Anton\Desktop\VK_voice\IEMOCAP_full_release_withoutVideos\Session'  # \dialog\EmoEvaluation'
    audio_dir_ = ''
    # r'C:\Users\Anton\Desktop\VK_voice\IEMOCAP_full_release_withoutVideos\Session'  # \dialog\wav'
    output_dir_ = ''
    # r'C:\Users\Anton\Desktop\VK_voice\voice_example'
    for i in range(1, 6):
        ann_d = annotations_dir_ + str(i) + r'\dialog\EmoEvaluation'
        aud_d = audio_dir_ + str(i) + r'\dialog\wav'
        # write_session(ann_d, aud_d, make_dir(output_dir_))
