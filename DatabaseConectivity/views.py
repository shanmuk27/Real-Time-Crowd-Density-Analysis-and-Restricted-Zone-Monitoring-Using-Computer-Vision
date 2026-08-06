from django.shortcuts import render

# Create your views here.
def Login(request):
    return render(request, 'login.html')
def Register(request):
    return render(request, 'register.html')
def Dashboard(request):
    return render(request, 'dashboard.html')
def ForgotPassword(request):
    return render(request, 'forgot-password.html')