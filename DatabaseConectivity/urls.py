from django.urls import path

from . import views

app_name = 'DatabaseConectivity'

urlpatterns = [
    path('', views.Home, name='home'),
    path('login/', views.Login, name='login'),
    path('register/', views.Register, name='register'),
    path('dashboard/', views.Dashboard, name='dashboard'),
    path('forgot-password/', views.ForgotPassword, name='forgot-password'),
    path('logout/', views.Logout, name='logout'),
    path('me/', views.Me, name='me'),
    path('users/', views.Users, name='users'),
    path('app/', views.App, name='app'),
    path('camera-management/', views.CameraManagement, name='camera-management'),
    path('profile/', views.Profile, name='profile'),
    path('profile/photo/', views.UploadPhoto, name='upload-photo'),
    path('profile/edit/', views.UpdateProfile, name='edit-profile'),
]
