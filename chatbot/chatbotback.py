import numpy as np
from abc import ABC, abstractmethod
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv('openAI.env')

# Interfaz para proveedor de IA
default_api_key = os.environ.get("OPENAI_API_KEY")
class AIProvider(ABC):
    @abstractmethod
    def get_response(self, texto, contexto, tools):
        pass

# Implementación concreta para OpenAI
class OpenAIProvider(AIProvider):
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = default_api_key
        self.client = OpenAI(api_key=api_key)

    def get_response(self, texto, contexto, tools):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=contexto,
            tools=tools,
            tool_choice="auto",
        )
        x = completion.choices[0].message.content
        if x is None:
            return "Esta vacio tu mensaje"
        else:
            return x

#Funcion de respuesta usando inversión de dependencias
def response(texto, contexto, tools, provider: AIProvider):
    return provider.get_response(texto, contexto, tools)

def answer_message(texto, contexto, tools, provider: AIProvider):
    return response(texto, contexto, tools, provider)
