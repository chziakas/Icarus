import sys
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import openai
import json
import os
from memory_constructor import MemoryConstructor
from dotenv import load_dotenv

load_dotenv()

class ImageCaptioner:
    def __init__(self):
        self.MEMORY_PROMPT = "Please formulate a detailed summary that captures the significance, ensuring it is memorable."
        self.AGENT_PROMPT = "Please describe the provided photo."
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.processor_deplot = Pix2StructProcessor.from_pretrained('google/deplot')
        self.model_deplot = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

    def caption_image_memory(self, image_path):
        image = Image.open(image_path)
        inputs_deplot = self.processor_deplot(images=image, text=self.MEMORY_PROMPT, return_tensors="pt")
        generated_ids_deplot = self.model_deplot.generate(**inputs_deplot, max_new_tokens=512)
        extracted_data = self.processor_deplot.decode(generated_ids_deplot[0], skip_special_tokens=True)
        print(extracted_data)
        memory_constructor = MemoryConstructor("gpt-3.5-turbo-16k", self.api_key)
        generated_memory = memory_constructor.generate(extracted_data)
        self._write_json(generated_memory, 'data/memory.json')
        return generated_memory

    def caption_image_action(self, image_path):
        image = Image.open(image_path)
        inputs_deplot = self.processor_deplot(images=image, text=self.AGENT_PROMPT, return_tensors="pt")
        generated_ids_deplot = self.model_deplot.generate(**inputs_deplot, max_new_tokens=512)
        extracted_data = self.processor_deplot.decode(generated_ids_deplot[0], skip_special_tokens=True)
        print(extracted_data)
        self._write_json(extracted_data, 'data/action.json')
        return extracted_data

    def _write_json(self, data: dict, filename:str):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":

    captioner = ImageCaptioner()

    image_path_memory = 'images/campfire.jpg'#sys.argv[1]

    response_memory = captioner.caption_image_memory(image_path_memory)
    print(response_memory)
    image_path_action = 'images/fire.jpg'#sys.argv[2]
    #response_action = captioner.caption_image_action(image_path_action)
    #print(response_action)
