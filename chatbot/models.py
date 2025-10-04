from django.db import models
from django.contrib.auth import get_user_model
from datetime import datetime
from django.utils import timezone
import numpy as np


def get_default_array():
  default_arr = np.random.rand(1536)  # Adjust this to your desired default array
  return default_arr.tobytes()


# Create your models here.
User = get_user_model()


class MedicalCenter(models.Model):
  name = models.CharField(max_length=255)
  specialty = models.CharField(max_length=255, null=True)
  adress = models.CharField(max_length=255)
  phone = models.IntegerField()
  schedule = models.CharField(max_length=255)
  emb = models.BinaryField(default=00)
  location=models.CharField(max_length=255,null=True)

class Specialist(models.Model):
  name = models.CharField(max_length=255)
  lastname = models.CharField(max_length=255)
  specialty = models.CharField(max_length=255)
  phone = models.IntegerField()
  location = models.CharField(max_length=255)
  schedule = models.CharField(max_length=255)
  embspecialist = models.BinaryField(default=b'\x00')
  id_medicalCenter = models.ForeignKey(MedicalCenter, on_delete=models.CASCADE)


class MedicalAppointment(models.Model):
  specialist = models.ForeignKey(Specialist, on_delete=models.CASCADE)
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  confirmation = models.CharField(max_length=255)


class Diagnosis(models.Model):
  diagnosis = models.TextField()
  recomendations = models.TextField()

class Chat(models.Model):
    id_chat = models.IntegerField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)

    # --- INICIO DEL CAMBIO: MÉTODO EN EL MODELO ---
    def get_first_message_content(self):
        # El 'related_name' por defecto es 'message_set'
        first_message = self.message_set.order_by('time').first()
        if first_message:
            return first_message.content
        return "Nuevo Chat"
    # --- FIN DEL CAMBIO ---

class Message(models.Model):
  def __str__(self) -> str:
    if self.user == None:
      return f'Author: bAImax : {self.content} '
    return f'Author: {self.user.first_name} {self.user.last_name} : {self.content} '
  user = models.ForeignKey(User , on_delete=models.CASCADE,blank = True, null = True)
  chat = models.ForeignKey(Chat, on_delete = models.CASCADE)
  content = models.TextField()
  time = models.DateTimeField(default=timezone.now)