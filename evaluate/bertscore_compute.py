import os
import sys
sys.path.append('/home/wuwl/project/msiip_medical_report_helper/medical_report_helper/data_compute/')
from utils import bertscore_compute

if __name__ == '__main__':

    data_type = 0  # xgj_MR=0 ; jz_CT=1
    device = 2

    if data_type == 0:
        data_type = 'xgj_MR'
    elif data_type == 1:
        data_type = 'jz_CT'

    # 定义相关路径
    parent_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'output', data_type))
    os.makedirs(parent_folder, exist_ok=True)

    bertscore_compute(parent_folder, device)

