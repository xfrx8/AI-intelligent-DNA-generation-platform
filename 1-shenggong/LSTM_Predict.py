import tensorflow as tf
import numpy as np
import pandas as pd
import os

class LSTM_Predict:
    def __init__(self, input_file_path, model_path='trained_model-n.h5'):
        self.model_path = model_path
        self.input_file_path = input_file_path
        self.model = self.load_model()

    @staticmethod
    def dna_sequence_to_one_hot(dna_sequence):
        mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        one_hot_sequence = np.array([mapping[base] for base in dna_sequence])
        return one_hot_sequence

    @staticmethod
    def load_txt_data(file_path):
        with open(file_path, 'r') as file:
            sequences = file.read().splitlines()
        one_hot_encoded_sequences = np.array([LSTM_Predict.dna_sequence_to_one_hot(seq) for seq in sequences])
        return sequences, one_hot_encoded_sequences

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径 {self.model_path} 不存在。请先训练并保存模型。")
        model = tf.keras.models.load_model(self.model_path)
        print("模型已加载。")
        return model

    def predict_and_save(self, output_path):
        if self.model is None:
            raise ValueError("模型未加载，请先加载模型。")
        
        sequences, one_hot_encoded_sequences = self.load_txt_data(self.input_file_path)
        predictions = self.model.predict(one_hot_encoded_sequences)

        # 对预测结果应用 exp(x) - 1 转换
        transformed_predictions = np.exp(predictions) - 1

        output_df = pd.DataFrame({
            'Sequence': sequences,
            'LSTM_Prediction': transformed_predictions.flatten()
        })
        output_df.to_csv(output_path, index=False)
        print(f"预测结果已保存到 {output_path}")

# # 使用示例
# input_file_path = 'input_sequences.txt'  # 替换为您的TXT文件路径
# predictor = LSTM_Predict(input_file_path)
# predictor.predict_and_save('output_predictions.csv')
