import torch
import os

data_type = 0  # xgj_MR=0 ; jz_CT=1
data_class = 3  # 1 , 2 , 3 , 4 , 'label_classifier' , 'instruction_test', 'describe_to_conclusion', 'both_way'

checkpoint_list = [4619, 9238, 13857, 18476, 23095, 27714, 32333, 36952, 41571, 46190]

for checkpoint in checkpoint_list:

    if data_type == 0:
        data_type_to_select = 'xgj_lora'
    elif data_type == 1:
        data_type_to_select = 'jz_lora'
    class_to_select = {1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 'label_classifier':'label_classifier', 'instruction_test':'instruction_test', 'describe_to_conclusion':'describe_to_conclusion', 'both_way':'both_way'}[data_class]

    # filename_1 = os.path.abspath(
    #     os.path.join(os.getcwd(), 'output', data_type_to_select, class_to_select, f'checkpoint-{checkpoint}',
    #                  'adapter_model.bin'))
    # filename_2 = os.path.abspath(
    #     os.path.join(os.getcwd(), 'output', data_type_to_select, class_to_select, 'adapter_model.bin'))

    filename_1 = os.path.abspath(
        os.path.join('/mnt/nvme_share/wuwl/project/ChatGLM-Tuning-master/output/xgj_lora/class_3/data_segmentation/split_5/', f'checkpoint-{checkpoint}', 'adapter_model.bin'))
    filename_2 = os.path.abspath(
        os.path.join('/mnt/nvme_share/wuwl/project/ChatGLM-Tuning-master/output/xgj_lora/class_3/data_segmentation/split_5/', 'adapter_model.bin'))

    adapters_weights_1 = torch.load(filename_1,
                                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    adapters_weights_2 = torch.load(filename_2,
                                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    keys_1 = adapters_weights_1.keys()
    keys_2 = adapters_weights_2.keys()

    key_list_1 = list(keys_1)
    key_list_2 = list(keys_2)


    def search_string_index(string_list, input_string):
        try:
            index = string_list.index(input_string)
            return index
        except ValueError:
            raise ValueError(f"未匹配到字符串序列")


    new_data = {}
    for key, value in adapters_weights_1.items():
        index = search_string_index(key_list_1, key)
        new_key = key_list_2[index]  # 替换键名
        new_data[new_key] = value
    torch.save(new_data, filename_1)

    adapters_weights_1 = torch.load(filename_1,
                                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    adapters_weights_2 = torch.load(filename_2,
                                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    keys_1 = adapters_weights_1.keys()
    keys_2 = adapters_weights_2.keys()

    key_list_1 = list(keys_1)
    key_list_2 = list(keys_2)

    for i in range(0, len(key_list_1)):
        if key_list_1[i] != key_list_2[i]:
            raise ValueError("checkpoint", checkpoint, "：key值匹配错误")
        else:
            pass

print("data_class", data_class, "：key值匹配正确")
