import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class DatasetModelMapper:
    def __init__(self):
        self.mappings = {
            'RAVDESS': {
                'Wav2Vec2_eng': {
                    4: 0,
                    6: 1,
                    5: 2,
                    2: 3,
                    3: 4
                },
                'HuBERT_eng': {
                    4: (0, 7),
                    6: (1, 8),
                    5: (2, 9),
                    2: (3, 10),
                    0: (4, 11),
                    3: (5, 12),
                    7: (6, 13)
                },
                'WavLM_eng': {}
            },
            'IEMOCAP': {
                'Wav2Vec2_eng': {
                    2: 0,
                    9: 1,
                    8: 2,
                    4: 3,
                    3: 4
                },
                'HuBERT_eng': {
                    2: (0, 7),
                    9: (1, 8),
                    8: (2, 9),
                    4: (3, 10),
                    0: (4, 11),
                    3: (5, 12),
                    6: (6, 13)
                },
                'WavLM_eng': {}
            }
        }
        self.class_names = {
            'RAVDESS': {
                'Wav2Vec2_eng': ["angry", "disgust", "fearful", "happy", "sad"],
                'HuBERT_eng': ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"],
                'WavLM_eng': []
            },
            'IEMOCAP': {
                'Wav2Vec2_eng': ["angry", "disgust", "fearful", "happy", "sad"],
                'HuBERT_eng': ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"],
                'WavLM_eng': []
            }
        }

    def mapping_df(self, dataset: str, model: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        label_mapping = self.mappings[dataset][model]
        df.labels = df.labels.apply(lambda x: label_mapping[x] if x in label_mapping.keys() else -1)
        return df.drop(df[df.labels == -1].index).reset_index(drop=True)

    def mapping_result(self, dataset: str, model: str, result_df: pd.DataFrame) -> pd.DataFrame:
        # здесь я забыл что значит оптимально
        result_df = result_df.copy()
        label_mapping = self.mappings[dataset][model]
        if model == 'HuBERT_eng':
            for i, _ in result_df.iterrows():
                o1 = 0
                o2 = 0
                for k, v in label_mapping.items():
                    if result_df.labels.iloc[i] == v and o1 == 0:
                        result_df.at[i, 'labels'] = k
                        o1 += 1
                    if result_df.predicted.iloc[i] in v and o2 == 0:
                        result_df.at[i, 'predicted'] = k
                        o2 += 1
        return result_df

    def metrics_with_classes(self, dataset: str, model: str, df: pd.DataFrame):
        if model == 'HuBERT_eng':
            df = self.mapping_result(dataset, model, df)
        else:
            df = df.copy()
        df.labels = df.labels.astype(int)
        df.predicted = df.predicted.astype(int)
        accuracy = accuracy_score(df.labels, df.predicted)
        f1_weighted = f1_score(df.labels, df.predicted, average='weighted')
        precision_weighted = precision_score(df.labels, df.predicted, average='weighted', zero_division=0)
        recall_weighted = recall_score(df.labels, df.predicted, average='weighted')

        f1_per_class = f1_score(df.labels, df.predicted, average=None)
        precision_per_class = precision_score(df.labels, df.predicted, average=None, zero_division=0)
        recall_per_class = recall_score(df.labels, df.predicted, average=None)
        print(f"{model} metrics on {dataset}")
        print("******")
        print(f"Accuracy: {round(accuracy,4)}")
        print(f"F1 Score: {round(f1_weighted, 4)}")
        print(f"Precision: {round(precision_weighted, 4)}")
        print(f"Recall: {round(recall_weighted, 4)}")

        print("\nMetrics per class:")
        for class_idx, (f1, prec, rec) in enumerate(zip(f1_per_class, precision_per_class, recall_per_class)):
            print(f"Class {self.class_names[dataset][model][class_idx]} - F1: {f1}, Precision: {prec}, Recall: {rec}")


if __name__ == "__main__":
    df_ = pd.DataFrame({'value': [-1, -1, -2, -3, -4, -5, -6, -7], 'labels': [0, 1, 2, 3, 4, 5, 6, 7]})
    mapped_df_1 = DatasetModelMapper().mapping_df(dataset='RAVDESS', model='Wav2Vec2_eng', df=df_)
    mapped_df_2 = DatasetModelMapper().mapping_df(dataset='RAVDESS', model='HuBERT_eng', df=df_)
    # print(mapped_df_2)
    df3 = pd.concat([mapped_df_2, pd.DataFrame({'predicted': [13,10,12,0,9,1,13]})], axis=1)
    df4 = pd.concat([mapped_df_1, pd.DataFrame({'predicted': [0,1,2,3,4]})], axis=1)
    # df3['mapped_labels'] = df3.apply(lambda x: transform_predictions(x.labels, x.predicted), axis=1)
    print(df3)
    print(DatasetModelMapper().metrics_with_classes('RAVDESS', 'HuBERT_eng', df3))
    # print(DatasetModelMapper().mapping_result('RAVDESS', 'HuBERT_eng', df3))




