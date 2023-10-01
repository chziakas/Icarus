import sys
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image


import openai
import json
import time
import os
from memory_constructor import MemoryConstructor

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MEMORY_PROMPT = "Please formulate a detailed summary that captures the significance, ensuring it is memorable."

AGENT_PROMPT = "Please formulate a detailed summary that captures the essential information."


def caption_image_memory(image_path):
    # Open the image
    image = Image.open(image_path)

    # Load the DePlot model and processor for extracting data from plots
    processor_deplot = Pix2StructProcessor.from_pretrained('google/deplot')
    model_deplot = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

    # Process the image and extract data using DePlot
    inputs_deplot = processor_deplot(images=image, text=MEMORY_PROMPT, return_tensors="pt")
    generated_ids_deplot = model_deplot.generate(**inputs_deplot, max_new_tokens=512)
    extracted_data = processor_deplot.decode(generated_ids_deplot[0], skip_special_tokens=True)
    print(extracted_data)
    memory_constructor = MemoryConstructor("gpt-3.5-turbo-16k", openai.api_key )
    generated_memory = memory_constructor.generate(extracted_data)
    write_json(generated_memory, 'data/memory.json')
    return generated_memory

def caption_image_action(image_path):
    # Open the image
    image = Image.open(image_path)

    # Load the DePlot model and processor for extracting data from plots
    processor_deplot = Pix2StructProcessor.from_pretrained('google/deplot')
    model_deplot = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

    # Process the image and extract data using DePlot
    inputs_deplot = processor_deplot(images=image, text=AGENT_PROMPT, return_tensors="pt")
    generated_ids_deplot = model_deplot.generate(**inputs_deplot, max_new_tokens=512)
    extracted_data = processor_deplot.decode(generated_ids_deplot[0], skip_special_tokens=True)
    print(extracted_data)
    write_json(extracted_data, 'data/action.json')
    return extracted_data


def write_json(data: dict, filename:str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python caption_image.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    response = caption_image_memory(image_path)
    print(response)
    
