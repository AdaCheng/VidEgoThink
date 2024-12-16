
import os
import json
import argparse
import datetime
import warnings
import torch
import numpy as np

import sys
import os
sys.path.append('./models')

from models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import disable_caching
disable_caching()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="BLIP2")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    # datasets
    parser.add_argument("--dataset_name", type=str, default=None, help='Your dataset name should be one element of the list as followed : [tg, og, fg, rm_critique, rm_feedback, assistance, navigation, hp_h2m, hp_m2l, vqa]')
    parser.add_argument('--annotation_path', type=str, required=True)
    # result_path
    parser.add_argument("--answer_path", type=str, default="./tiny_answers")
    parser.add_argument("--data_mode", type=str, default=None, help="You can chooce 'image' or 'video' type of your data")
    parser.add_argument("--video_folder", type=str, default="/apdcephfs_cq10/share_1150325/csj/videgothink/goalstep_val_hp_video")
    parser.add_argument("--image_folder", type=str, default=None)

    # parser.add_argument("--planning", action="store_true")

    args = parser.parse_args()
    return args

def load_dataset(args):
    # 确保文件夹存在
    folder_check = {
        'image': args.image_folder,
        'video': args.video_folder
    }
    folder_path = folder_check.get(args.data_mode)
    if folder_path and not os.path.exists(str(folder_path)):
        raise FileNotFoundError(f"{args.data_mode.capitalize()} folder '{folder_path}' does not exist.")
    with open(args.annotation_path, 'r') as f:
        dataset = json.load(f)

    for i, d in enumerate(dataset):
        if args.data_mode == 'image':
            image_filename = d.get('image_path', [''])[0]
            # print(f"diffeerence : {d.get('image_path', [''])[0]} and {d.get('image_path', [''])}")
            if not image_filename:
                warnings.warn(f"No image found in dataset entry {i}.", UserWarning)
            dataset[i]['images'] = os.path.join(args.image_folder, image_filename)
        elif args.data_mode == 'video':
            video_filename = d.get('video_path', '')
            if not video_filename:
                warnings.warn(f"No video found in dataset entry {i}.", UserWarning)
            dataset[i]['video'] = os.path.join(args.video_folder, video_filename)
    return dataset


def get_generation_args(dataset_name):
    print(f"---------function dataset name {dataset_name}------------")
    # 基础参数模板
    base_args = {'max_new_tokens': 300}
    # 定义不同数据集的特定配置
    dataset_configs = {
        "tg": {'max_new_tokens': 128, 'tg': True},
        "og": {'max_new_tokens': 128, 'og': True},
        "fg": {'max_new_tokens': 128, 'fg': True},
        "rm_critique": {'max_new_tokens': 128, 'rm_critique': True},
        "rm_feedback": {'max_new_tokens': 128, 'rm_feedback': True},
        "assistance": {'max_new_tokens': 300, 'planning': True},
        "navigation": {'max_new_tokens': 300, 'planning': True},
        "hp_h2m": {'max_new_tokens': 300, 'hp_h2m': True},
        "hp_m2l": {'max_new_tokens': 300, 'hp_m2l': True},
        "vqa":{'max_new_tokens': 300}
    }
    # 获取并合并配置
    config = dataset_configs.get(dataset_name)
    if config is None:
        # 如果没有找到匹配的配置，警告
        warnings.warn(f"Warning: No specific configuration found for dataset '{dataset_name}'. Using base configuration.")
        config = base_args
    return {**base_args, **config}

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_names = args.model_name.split(',')  
    time = datetime.datetime.now().strftime("%m%d-%H%M")
    dataset_name = args.dataset_name

    print(f"dataset name: {dataset_name}")
    print(f"Running inference on {dataset_name}")

    for model_name in model_names:
        print(f"Running inference: {model_name}")

        if 'blip2' in model_name.lower() or 'llava' in model_name.lower():
            batch_size = 1
        else:
            batch_size = args.batch_size
        device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        model = get_model(model_name, device=device)
        
        dataset = load_dataset(args)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        model_answers = []
        ref_answers = []
        question_files = []
        q_id = 0
        for batch in tqdm(dataloader, desc=f"Running inference: {model_name} on  {dataset_name}"):
            questions = batch['question']

            input_media = {
                'image': batch.get('images', None),
                'video': batch.get('video', None)
            }.get(args.data_mode, None) # choose the right type of data mode
            print(f"args.data_mode:{args.data_mode} , dataset_name: {dataset_name}, media: {input_media}")  

            if args.batch_size == 1:
                try:
                    output = model.generate(input_media[0], questions[0], **get_generation_args(dataset_name))
                    print(output)
                    outputs = [output]
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error: {e}")
                    outputs = ['']
            else:
                outputs = model.batch_generate(input_media, questions, **get_generation_args(dataset_name))
            for i, (question, answer, pred) in enumerate(zip(batch['question'], batch['answer'], outputs)):
                # answer_dict={'question': questions, 'prediction': pred,
                # 'gt_answers': answer}
                # model_answers.append(answer_dict)
                model_answers.append({
                    'question_id': q_id,
                    'model_id': model_name,
                    'choices':[{'index': 0, "turns": [pred]}]
                })
                ref_answers.append({
                    'question_id': q_id,
                    'model_id': 'ground_truth',
                    'choices':[{'index': 0, "turns": [answer]}]
                })
                question_files.append({
                    'question_id': q_id,
                    'turns': [question]
                })
                q_id += 1
        torch.cuda.empty_cache()
        del model



        result_folder = args.answer_path
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        model_answer_folder = os.path.join(result_folder, 'model_answer')
        if not os.path.exists(model_answer_folder):
            os.makedirs(model_answer_folder)
        with open(os.path.join(model_answer_folder, f"{model_name}.jsonl"), 'w') as f:
            for pred in model_answers:
                f.write(json.dumps(pred) + '\n')
        
        ref_answer_folder = os.path.join(result_folder, 'reference_answer')
        if not os.path.exists(ref_answer_folder):
            os.makedirs(ref_answer_folder)
        with open(os.path.join(ref_answer_folder, "ground_truth.jsonl"), 'w') as f:
            for ref in ref_answers:
                f.write(json.dumps(ref) + '\n')
        
        with open(os.path.join(result_folder,  "question.jsonl"), 'w') as f:
            for q in question_files:
                f.write(json.dumps(q) + '\n')

        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
