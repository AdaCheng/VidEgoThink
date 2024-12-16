from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
import cv2
import os
import io
import base64


class TestQwenVLChat:
    def __init__(self, device=None):
        torch.manual_seed(1234)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("/disks/disk1/share/models/Qwen-VL-Chat", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("/disks/disk1/share/models/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained("/disks/disk1/share/models/Qwen-VL-Chat", trust_remote_code=True)

    def generate_img(self, image, question, max_new_tokens=256, planning=False):
        query = self.tokenizer.from_list_format([
            {'image': image},
            {'text': question},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response
    
    # def generate_vid(self, image, question, max_new_tokens=256, planning=False):
    #     img = Image.open(image)
    #     byte_io = io.BytesIO()
    #     img.save(byte_io, format='JPEG')
    #     byte_io.seek(0)
    #     img_bytes = byte_io.read()
    #     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    #     query = self.tokenizer.from_list_format([
    #         {'image': f"data:image/jpeg;base64,{img_base64}"},
    #         {'text': question},
    #     ])
    #     response, history = self.model.chat(self.tokenizer, query=query, history=None)
    #     return response

    def generate(self, image, question, max_new_tokens=256, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
        if rm_critique:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif rm_feedback:
            question = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif hp_h2m:
            question = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        else:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)
            
        query = self.tokenizer.from_list_format([
            {'image': image},
            {'text': question},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def generate_vid(self, image, question, max_new_tokens=256, planning=False):
        img = Image.open(image)
        byte_io = io.BytesIO()
        img.save(byte_io, format='JPEG')
        byte_io.seek(0)
        print(self.tokenizer.from_list_format.__module__, self.tokenizer.from_list_format.__name__)
        query = self.tokenizer.from_list_format([
            {'image': f"data:image/jpeg;base64,{base64.b64encode(byte_io.read()).decode('utf-8')}"},
            {'text': question},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response
    
    def generate_grounding(self, image, question, max_new_tokens=256, planning=False):
        query = self.tokenizer.from_list_format([
            {'image': image},
            {'text': "框出图中击掌的位置"},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestQwenVLChat(device)
    image = "/disks/disk1/share/EgoThink/data/vqa/video/34.mp4"
    question = "describe the image"
    response = model.generate(image, question)
    print(response)