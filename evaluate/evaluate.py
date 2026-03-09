import pandas as pd
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
from metrics import compute_scores
import jieba
import os
import codecs
# 设置 JAVA_HOME 环境变量为你的 Java 安装路径
os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk-1.8'
# 将 Java bin 目录添加到 PATH 环境变量
os.environ['PATH'] = os.environ['JAVA_HOME'] + '\\bin;' + os.environ['PATH']
metric_ftns = compute_scores
file_path = 'D:/project/msiip/msiip_medical_report_helper/output/xgj_MR'


project_name_list = ['CLSFT_test']

for project_name in project_name_list:
    print('')
    print('*****',project_name,'*****')
    file_name = os.path.join(file_path, project_name)
    file_output = os.path.join(file_name, 'score_final.txt')
    refs = os.path.join(file_name, 'test.jsonl.refs')

    train_set = open(refs, 'r', encoding='utf-8').read().splitlines()


    for i in range(4,11):
        hyps = os.path.join(file_name, f'test.jsonl{str(i)}.hyps')
        try:
            test_set = open(hyps, 'r', encoding='utf-8').read().splitlines()
            if len(test_set) != len(train_set):
                continue
            print('epoch:', i)
            train_met = metric_ftns({i: [' '.join(jieba.cut(gt))] for i, gt in enumerate(train_set)},
                                        {i: [' '.join(jieba.cut(re))] for i, re in enumerate(test_set)})

            print(train_met.items())
            output_list = []
            output = ''
            for item in train_met.items():
                for element in item:
                    output += str(element)
                    output += '   '
                output += '|   '
            output_list.append(output)
            with codecs.open(file_output, mode='a', encoding='utf-8') as f:
                pd.DataFrame(output_list).to_csv(f, sep='\t', header=False, index=False)

        except FileNotFoundError:
            pass
