from django.contrib import admin
from .models import *
# Register your models here. 
admin.site.register(MedicalCenter)
admin.site.register(MedicalAppointment)
admin.site.register(Specialist)
admin.site.register(Diagnosis)
admin.site.register(Message)
admin.site.register(Chat)