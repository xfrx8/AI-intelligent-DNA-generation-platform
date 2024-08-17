from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS
import os
import csv
import math

app = Flask(__name__, static_folder='dist')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DOWNLOAD_FOLDER'] = 'downloads/'

# 确保上传和下载目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# 导入模型类
from TF_Predict import TF_Predict
from CNN_Predict import CNN_Predict
from LSTM_Predict import LSTM_Predict

# 初始化模型实例
models = {
    'tf': TF_Predict,
    'cnn': CNN_Predict,
    'lstm': LSTM_Predict,
}

def trim_sequence(sequence):
    """裁减序列，只保留前29个字符"""
    return sequence[:29]

def delete_uploaded_files(folder):
    """删除上传文件夹中的所有文件"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'删除文件 {file_path} 失败. 原因: {e}')

def delete_downloaded_files(folder):
    """删除下载文件夹中的所有文件"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'删除文件 {file_path} 失败. 原因: {e}')

@app.route('/')
def index():
    # 清空上传和下载目录
    delete_uploaded_files(app.config['UPLOAD_FOLDER'])
    delete_downloaded_files(app.config['DOWNLOAD_FOLDER'])
    return send_from_directory(app.static_folder, 'index.html')
@app.route('/<path:path>')
def static_proxy(path):
    # send_static_file will guess the correct MIME type
    return app.send_static_file(path)
@app.route('/submit', methods=['POST'])
def submit():
    project_name = request.form.get('project_name')
    sequence = request.form.get('sequence')
    file = request.files.get('file')

    # 每次提交新的TXT文件前清空上传和下载目录
    delete_uploaded_files(app.config['UPLOAD_FOLDER'])
    delete_downloaded_files(app.config['DOWNLOAD_FOLDER'])

    original_sequences = []
    if sequence:
        original_sequences.append(sequence)
        trimmed_sequence = trim_sequence(sequence)
        filepath = save_sequence_to_file(trimmed_sequence)
    elif file and file.filename.endswith('.txt'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        original_sequences = read_sequences_from_file(filepath)
        trimmed_sequences = [trim_sequence(seq) for seq in original_sequences]
        filepath = save_sequences_to_file(trimmed_sequences)
    else:
        return '未提供有效输入或序列长度不正确'

    output_files = process_file(filepath, original_sequences, project_name)

    return render_template('index.html', output_files=output_files)

def save_sequence_to_file(sequence):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input_sequence.txt')
    with open(filepath, 'w') as f:
        f.write(sequence)
    return filepath

def read_sequences_from_file(filepath):
    with open(filepath, 'r') as f:
        sequences = [line.strip() for line in f.readlines()]
    return sequences

def save_sequences_to_file(sequences):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'trimmed_sequences.txt')
    with open(filepath, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')
    return filepath

def process_file(filepath, original_sequences, project_name):
    output_files = []
    for model_name, model_class in models.items():
        output_filename = f'{model_name}_predictions.csv'
        output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
        model = model_class(filepath)
        model.predict_and_save(output_path)
        update_csv(output_path, original_sequences, project_name)
        output_files.append(output_filename)
    return output_files

def update_csv(csv_path, original_sequences, project_name):
    temp_csv_path = csv_path + '.tmp'
    with open(csv_path, 'r') as infile, open(temp_csv_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        headers = next(reader)
        writer.writerow([f'Project name: {project_name}'])
        writer.writerow(headers)
        for idx, row in enumerate(reader):
            if idx < len(original_sequences):
                row[0] = original_sequences[idx]
            writer.writerow(row)
    os.replace(temp_csv_path, csv_path)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
