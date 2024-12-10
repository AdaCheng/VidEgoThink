import os
import json
import argparse
import datetime
from tqdm import tqdm

from openai import AzureOpenAI
from openai import OpenAI
import openai
import requests
import base64
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from m2l_prompt import m2l_caption_prompt, m2l_frame_prompt, m2l_text_prompt


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class LLM_API:
    def __init__(self, model, base_url=None, temperature=0.0, stop=None):
        self.model = model
        self.temperature = temperature
        self.n_repeat = 1
        self.stop = stop

        if "gpt" in model.lower():
            self.api_key = ""
            self.client = AzureOpenAI(
                    api_key=self.api_key,  
                    api_version="2024-02-01",
                    azure_endpoint = ""
                    )
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            assert base_url is not None
            self.client = OpenAI(api_key="ss", base_url=base_url)

    def request_general(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=self.temperature,
            stop=self.stop,
        )
        return response.choices[0].message.content
    
    def request_vision(self, img_dir, prompt):
        vision_messages = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image)}"
            }
        } for image in img_dir]

        content = [{
            "type": "text",
            "text": prompt,
        }]
        for message in vision_messages:
            content.append(message)
        all_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            stream=False,
            temperature=self.temperature,
            stop=self.stop,
        )
        return response.choices[0].message.content

def load_dataset(args):
    annotation_path = args.annotation_path
    with open(annotation_path, 'r') as f:
        dataset = json.load(f)
    for i, d in enumerate(dataset):
        video_file = d['video_path']
        image_files = d['image_path']
        if args.inference_type == "caption":
            captions = d['caption']
        dataset[i]['video'] = os.path.join(args.video_folder, video_file)
        dataset[i]['images'] = [os.path.join(args.image_folder, image_file) for image_file in image_files]
    return dataset

def generate_item(args, item):  
    q_id = item["q_id"]
    question = item["question"]
    images = item["images"]
    answer = item["answer"]
    if args.inference_type == "caption":
        caption = item["caption"]
        if "vqa" in args.task:
            prompt = f"Imagine you are the camera wearer (I) who recorded the video.\nHere is the captions of the video:\n{caption}.\nPlease directly answer the question as short as possible.\nQuestion: {question} Short answer:"
        elif args.task == "hp_high2mid":
            prompt = f"Imagine you are the camera wearer (I) who recorded the video.\nHere is the captions of the video: {caption}.\n\nGiven the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {question} Short answer:"
        elif args.task == "hp_mid2low":
            prompt = m2l_caption_prompt + f"\n\Here is the caption of the video: {caption}.\nQuestion: {question}\nList of actionable functions:"
        elif args.task == "rm_critique":
            prompt = f"Imagine you are the camera wearer (I) who recorded the video.\nHere is the captions of the video:\n{caption}.\n Please directly answer yes or no to determin whether the task is completed or not. Question: {question} Short answer:"
        elif args.task == "rm_feedback":
            prompt = f"Imagine you are the camera wearer (I) who recorded the video.\nHere is the captions of the video:\n{caption}.\nThe video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible.\nQuestion: {question}\nShort Answer:"
    elif "frames" in args.inference_type:
        if args.task == "rm_critique":
            prompt = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif args.task == "rm_feedback":
            prompt = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif args.task == "hp_high2mid":
            prompt = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        elif "vqa" in args.task:
            prompt = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)
        elif args.task == "hp_mid2low":
            prompt = m2l_frame_prompt + "{question} List of actionable functions:"
    elif args.inference_type in ["narration", "text"]:
        if args.task == "rm_critique":
            prompt = "Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif args.task == "rm_feedback":
            prompt = "Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif args.task == "hp_high2mid":
            prompt = "Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        elif "vqa" in args.task:
            prompt = "Please directly answer the question as short as possible. Question: {} Short answer:".format(question)
        elif args.task == "hp_mid2low":
            prompt = m2l_text_prompt + "{question} List of actionable functions:"

    max_retries = 5  # 最大重试次数
    retry_delay = 2  # 重试之间的延时，单位为秒
    attempt = 0  # 当前尝试次数
    if "frames" in args.inference_type:
        while True:
            try:
                output = llm_api.request_vision(images, prompt)
                if "Short Answer: " in output:
                    output = output.split("Short Answer: ")[1]
                print(output)
                break
            except Exception as e:
                # print(e)
                if attempt >= max_retries:
                    print(e)
                    output = "error."
                    break
                time.sleep(retry_delay)
                attempt += 1
    else:
        while True:
            try:
                output = llm_api.request_general(prompt)
                if "Short Answer: " in output:
                    output = output.split("Short Answer: ")[1]
                print(output)
                break
            except Exception as e:
                # print(e)
                attempt += 1
                if attempt >= max_retries:
                    print(e) 
                    output = "error."
                    print(output)
                    break
    return output, question, answer, q_id



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT Inference on a dataset")

    # models
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--inference_type", type=str, default="frames")

    # datasets
    parser.add_argument('--annotation_path', type=str, default="/apdcephfs_cq10/share_1150325/csj/videgothink/final_goalstep_rm_critique.json")
    parser.add_argument('--video_folder', type=str, default="/apdcephfs_cq10/share_1150325/csj/videgothink/goalstep_val_clean/")
    parser.add_argument('--image_folder', type=str, default="/apdcephfs_cq10/share_1150325/csj/videgothink/goalstep_val_rm_keyframe/")
    parser.add_argument("--answer_path", type=str, default="./answer/rm_critique")
    parser.add_argument('--task', type=str, default="rm_critique")

    args = parser.parse_args()

    llm_api = LLM_API(args.model_name)

    dataset = load_dataset(args)
    for i, item in enumerate(dataset):
        item["q_id"] = i + 1
    model_answers = []
    ref_answers = []
    question_files = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_item = {executor.submit(generate_item, args, item): item for item in dataset}
            # future = submit_with_retry(executor, generate_item, args, item)
            # future_to_item[future] = item
    
        # 等待每个任务完成并处理结果
        for future in tqdm(as_completed(future_to_item),  total=len(future_to_item), desc=f"Running {args.model_name} on task {args.task}"):
            item = future_to_item[future]
            try:
                output, question, answer, q_id  = future.result()
                print(question)
            except Exception as e:
                print(f"处理项目 {item} 时发生错误: {e}")

            model_answers.append({
                "question_id" : q_id,
                "model_id" : args.model_name,
                "choices" : [{"index" : 0, "turns" : [output]}]
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
    
    result_folder = args.answer_path
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model_answer_folder = os.path.join(result_folder, 'model_answer')
    if not os.path.exists(model_answer_folder):
        os.makedirs(model_answer_folder)
    with open(os.path.join(model_answer_folder, f"{args.model_name}-{args.inference_type}.jsonl"), 'w') as f:
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