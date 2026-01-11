import sys
import json 
import random 
from openai import OpenAI

class deepseek_v3():
    def __init__(self, model_name="deepseek-v3", temperature=0) -> None:
        # deepseek_v3
        self.api_key = ""
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        self.temperature = temperature
        print(f"model_name: {self.model_name}; temperature:{self.temperature}")

    
    def __call__(self, message, maxtry=3):
        assert isinstance(message, str), 'The input prompt for cfbench should be a string.'
        messages = [{"role":"user", "content": message}]
        i = 0
        response = "N/A"
        while i < maxtry:
            try:
                if self.temperature is None:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages,
                        temperature=self.temperature
                    )
                response = response.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t message:{message} \tError:{e}", flush=True)
                i += 1
                continue
        return response

if __name__ == "__main__":
    print(deepseek_v3()("1+1"))
    
    
