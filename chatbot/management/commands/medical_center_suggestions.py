import chatbot as chatbot
from chatbot import models
from chatbot.models import MedicalCenter
from django.core.management.base import BaseCommand
import os
import numpy as np
from openai import OpenAI


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv('openAI.env')
client = OpenAI(

    api_key=os.environ.get('openAI_api_key'),
)



def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


#Sugerir centro medico basandonos en su especialidad. Los emb de los medical centers estan basados en el campo speciality
#Se migrara el comando a forma de script para su posterior uso de forma normal. 

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        help = 'Modify path of images'

        promp="Orthopedic Surgery and Sports Medicine"

        embpromp=get_embedding(promp)
        
        centros_medicos=MedicalCenter.objects.all()
        lista=[]
        for centro in centros_medicos:
            item=centro.emb
            item=list(np.frombuffer(item))
            lista.append(cosine_similarity(item,embpromp))

        lista = np.array(lista)
        idx = np.argmax(lista)
        idx = int(idx)
        print(centros_medicos[idx].name)

        self.stdout.write(self.style.SUCCESS(f'Solicitud hecha de forma exitosa')) 

