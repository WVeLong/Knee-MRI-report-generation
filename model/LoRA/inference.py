from transformers import AutoModel
import torch
from transformers import AutoTokenizer
from peft import PeftModel
import json
from cover_alpaca2jsonl import format_example
from tqdm import tqdm
import os
import csv
import codecs
import pandas as pd

def inference(instructions, start_index, end_index, file_path_output):
    with torch.no_grad():
        max_retry = 5
        for idx, item in tqdm(enumerate(instructions[start_index:end_index]), desc="当前进度"):
            feature = format_example(item)
            input_text = feature['context']
            ids = tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids])
            input_ids = input_ids.to('cuda')  # 将输入数据移动到 GPU 并转换为半精度
            retry_count = 0
            while retry_count < max_retry:
                out = model.generate(
                    input_ids=input_ids,
                    max_length=350,
                    do_sample=False,
                    temperature=0.1,
                    num_beams=1,  # 设置beam数量为3，启用Beam Search
                    early_stopping=True  # 当所有beam都找到EOS标记时停止生成
                )
                out_text = tokenizer.decode(out[0])
                answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
                try:
                    cut_index = answer.index("Answer: 征象描述:")
                except ValueError:
                    try:
                        cut_index = answer.index("Answer: 征象显示")
                    except ValueError:
                        try:
                            cut_index = answer.index("征象描述")
                        except ValueError:
                            retry_count += 1
                            continue
                break  # 如果成功匹配到条件，则跳出循环

            if retry_count == max_retry:
                print(answer)
                raise ValueError("Neither '诊断结论' nor '征象描述' found in the answer.")

            prefix = "Answer: "
            input_string = answer[cut_index:].strip()
            if input_string.startswith(prefix):
                result = input_string[len(prefix):]
            else:
                result = input_string
            # print(result)
            output = [result]
            with open(file_path_output, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')  # 使用制表符作为分隔符
                writer.writerow(output)  # 将 output 作为单元素列表写入，确保它作为一个字段被写入


if __name__ == "__main__":

    data_type = 0      # xgj_MR=0 ; jz_CT=1
    data_class = '3'     # 1 , 2 , 3 , 4 , 'label_classifier' , 'instruction_test', 'describe_to_conclusion' ,'both_way'
    epoch = 9       # 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10


    start_index = 0
    end_index = 105

    checkpoint_list = [11547, 23094, 34641, 46188, 57735, 69282, 80829, 92376, 103923, 115470]

    checkpoint =  checkpoint_list[epoch - 1]
    output = 'test.jsonl' + str(epoch) + '.hyps'

    if data_type == 0:
        data_type_to_select = 'xgj_lora'
    elif data_type == 1:
        data_type_to_select = 'jz_lora'
    # class_to_select = {1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 'label_classifier':'label_classifier', 'instruction_test':'instruction_test', 'describe_to_conclusion':'describe_to_conclusion', 'both_way':'both_way'}[data_class]

    model = AutoModel.from_pretrained(
        "/mnt/nvme_share/wuwl/chatglm-6B/",
        trust_remote_code=True,
        load_in_8bit=True,
        device_map='auto',
        torch_dtype=torch.float16,
        output_hidden_states=True,
        output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme_share/wuwl/chatglm-6B/", trust_remote_code=True)
    lora_weights = os.path.abspath(os.path.join('/mnt/nvme_share/wuwl/project/ChatGLM-Tuning-master/output/xgj_lora/class_2/both_way/', f'checkpoint-{checkpoint}'))
    file_path_output = os.path.abspath(os.path.join('/mnt/nvme_share/wuwl/project/ChatGLM-Tuning-master/output/xgj_lora/集外/青医附院/', output))
    file_path_instructions = os.path.abspath('/mnt/nvme_share/wuwl/project/ChatGLM-Tuning-master/output/xgj_lora/集外/青医附院/修改_青医附院_inference.json')
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    #file_path_instructions = os.path.abspath(os.path.join(os.getcwd(), 'output', data_type_to_select, class_to_select, 'inference', 'test.json')
    instructions = json.load(open(file_path_instructions))


    if os.path.exists(file_path_output):
        history = open(file_path_output, 'r', encoding='utf-8').read().splitlines()
        if len(history) <= end_index:
            start_index = len(history)

    print("data_class", data_class, " epoch", epoch, " 开始推理")
    print("当前剩余item数量为：", end_index - start_index)
    inference(instructions, start_index, end_index, file_path_output)
    print("data_class", data_class, " epoch", epoch, " 推理完成")
