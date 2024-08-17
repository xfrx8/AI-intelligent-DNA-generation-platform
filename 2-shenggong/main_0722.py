from flask import Flask, request, render_template, jsonify, send_file
from flask import Flask, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import datetime
import json

app = Flask(__name__, static_folder='dist')
CORS(app)

# 保存class CSV文件的目录路径
csv_dir = "final_data"
output_dir = "output_files"
json_dir = "json_files"

# 确保CSV目录、输出目录和JSON目录存在
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')
@app.route('/<path:path>')
def static_proxy(path):
    # send_static_file will guess the correct MIME type
    return app.send_static_file(path)
@app.route('/submit', methods=['POST'])
def submit():
    try:
        project_name = request.form['project_name']
        strength_class = request.form['strength_class']
        promoter_numbers = int(request.form['promoter_numbers'])

        # 检查强度类是否在允许的范围内
        if not strength_class.isdigit() or not 1 <= int(strength_class) <= 8:
            return jsonify({"error": "请输入1-8之间的强度"}), 400

        # 构建文件名
        csv_file = os.path.join(csv_dir, f"class_{strength_class}.csv")

        # 检查CSV文件是否存在
        if not os.path.isfile(csv_file):
            return jsonify({"error": "强度文件不存在"}), 400

        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 检查promoter_numbers是否超过文件行数
        if promoter_numbers < 1 or promoter_numbers > 100:
            return jsonify({"error": "请输入1-100之间的一个数"}), 400

        # 检查是否有score列
        if 'Score' not in df.columns:
            return jsonify({"error": "CSV文件中缺少 'Score' 列"}), 400

        # 按分数段抽取数据
        samples = {
            "100": df[df['Score'] == 100],
            "90-99": df[(df['Score'] >= 90) & (df['Score'] <= 99)],
            "70-89": df[(df['Score'] >= 70) & (df['Score'] <= 89)],
            "0-69": df[df['Score'] < 70],
        }

        counts = {
            "100": int(promoter_numbers * 0.4),
            "90-99": int(promoter_numbers * 0.2),
            "70-89": int(promoter_numbers * 0.2),
            "0-69": int(promoter_numbers * 0.2),
        }

        result_df = pd.DataFrame()
        remaining_df = df.copy()

        for key, sample_df in samples.items():
            needed_samples = counts[key]
            if len(sample_df) < needed_samples:
                # 如果当前分数段的样本不足，则抽取全部，并从未被抽取的样本中随机补足
                result_df = pd.concat([result_df, sample_df])
                remaining_df = remaining_df.drop(sample_df.index, errors='ignore')
                shortage = needed_samples - len(sample_df)
                if shortage > 0:
                    additional_samples = remaining_df.sample(n=shortage)
                    result_df = pd.concat([result_df, additional_samples])
                    remaining_df = remaining_df.drop(additional_samples.index, errors='ignore')
            else:
                # 如果当前分数段的样本足够，则按需求数量抽取
                selected_samples = sample_df.sample(n=needed_samples)
                result_df = pd.concat([result_df, selected_samples])
                remaining_df = remaining_df.drop(selected_samples.index, errors='ignore')

        # 随机打乱结果中的序列顺序
        result_df = result_df.sample(frac=1).reset_index(drop=True)

        # 生成新的CSV文件名，附加时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_file = f"class_{strength_class}_{promoter_numbers}_{timestamp}.csv"
        output_csv_path = os.path.join(output_dir, output_csv_file)

        # 添加项目名称行
        with open(output_csv_path, 'w') as f:
            f.write(f"Project name: {project_name}\n")
            result_df.to_csv(f, index=False)

        # 抽取分数最高的10个序列
        top_sequences = result_df.nlargest(10, 'Score') if len(result_df) >= 10 else result_df

        # 构建返回的JSON数据
        response_data = {
            "filename": output_csv_file,
            "top_sequences": top_sequences.to_dict(orient='records')
        }

        # 保存JSON数据到文件
        json_file = f"result.json"
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'w') as f:
            json.dump(response_data, f, indent=4)

        return jsonify(response_data)
    except ValueError:
        return jsonify({"error": "输入无效。请确保启动子数量是有效的整数。"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    try:
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "文件不存在。"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
