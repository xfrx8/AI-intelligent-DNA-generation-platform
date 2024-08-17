import numpy as np
import tensorflow as tf
import csv

class CNN_Predict:
    def __init__(self,input_file, model_path = 'weight_MSE_c.v-n.h5' ):
        self.model_path = model_path
        self.input_file = input_file
        self.model = self.load_model()
        self.sequences = self.load_sequences()

    def load_model(self):
        model = tf.keras.models.load_model(self.model_path)
        print("模型已加载。")
        return model

    def load_sequences(self):
        with open(self.input_file, 'r') as f:
            sequences = f.read().splitlines()
        return sequences

    def seq2onehot(self, seq):
        mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
        return np.array([[mapping[base] for base in s] for s in seq])

    def predict_and_save(self, output_file='downloads/cnn_predictions.csv'):
        sequences_onehot = self.seq2onehot(self.sequences)
        sequences_onehot = sequences_onehot.reshape(-1, 29, 1, 4)  # 调整形状以适应模型输入
        predictions = self.model.predict(sequences_onehot, verbose=1)
        
        # 对预测结果应用 exp(x) - 1 转换
        transformed_predictions = np.exp(predictions) - 1
        
        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Sequence', 'CNN_Prediction'])
            for seq, pred in zip(self.sequences, transformed_predictions):
                csvwriter.writerow([seq, pred[0]])
        print(f"预测结果已保存到 {output_file}。")

# # 使用方法
# input_file = 'input_sequences.txt'
# predictor = CNN_Predict( input_file)
# predictor.predict_and_save()
