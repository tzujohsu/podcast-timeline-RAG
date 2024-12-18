import os
from tqdm import tqdm

from utils import restore_punctuation
from transformers import AutoModelForTokenClassification,AutoTokenizer

from opencc import OpenCC

PATH = 'data'
raw_list = os.listdir(PATH+'/rawtext')

if not os.path.exists(PATH + '/punc_restored'):
    os.makedirs(PATH + 'punc_restored')

restored_list = os.listdir(PATH + '/punc_restored')
todolist = list(set(raw_list) - set(restored_list))

cc = OpenCC('s2twp')

for path in tqdm(todolist):
    # read raw text
    with open(f'{PATH}/rawtext/{path}', 'r') as f:
        raw_text = f.read()

    restored_text = restore_punctuation(raw_text)
    tc_converted = cc.convert(restored_text)

    with open(f'{PATH}/punc_restored/{path}', 'w') as file:
        file.write(tc_converted)
    
    