from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chatbot/<int:room>/',views.chatbot,name='chatbot'),
    path('checkview',views.checkview,name='checkview'),
    path('login/', views.login_, name="login"),
    path('signup/',views.signup, name="signup"),
    path('send', views.send, name='send'),
    path('getMessages/<int:chatid>/', views.getMessages, name='getMessages'),
    path('logout/', views.logout_, name="logout"),
    path('chats/',views.chat_page,name='chat_page'),
    path('emergency', views.emergency, name = 'emergency')
]