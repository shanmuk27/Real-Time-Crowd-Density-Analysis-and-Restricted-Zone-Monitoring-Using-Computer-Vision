import json

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import users


def _json_body(request):
    try:
        return json.loads(request.body or b'{}')
    except (ValueError, TypeError):
        return {}


def _user_dict(user):
    return {'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role}


@ensure_csrf_cookie
def Home(request):
    user_id = request.session.get('user_id')
    if user_id and users.objects.filter(id=user_id).exists():
        return redirect('DatabaseConectivity:dashboard')
    return redirect('DatabaseConectivity:login')


def Login(request):
    if request.method == 'POST':
        data = _json_body(request)
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        try:
            user = users.objects.get(email=email)
        except users.DoesNotExist:
            return JsonResponse({'ok': False, 'message': 'No account found for this email. Please register first.'})
        if user.password != password:
            return JsonResponse({'ok': False, 'message': 'Incorrect password. Please try again.'})
        request.session.flush()
        request.session['user_id'] = user.id
        request.session['role'] = user.role
        return JsonResponse({'ok': True, 'user': _user_dict(user)})
    return render(request, 'login.html')


@ensure_csrf_cookie
def Register(request):
    if request.method == 'POST':
        data = _json_body(request)
        name = (data.get('name') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        role = data.get('role') or ''
        if not name or not email or not password or not role:
            return JsonResponse({'ok': False, 'message': 'All fields are required.'})
        if users.objects.filter(email=email).exists():
            return JsonResponse({'ok': False, 'message': 'An account with this email already exists. Try logging in instead.'})
        user = users.objects.create(name=name, email=email, password=password, role=role, control='full')
        request.session.flush()
        request.session['user_id'] = user.id
        request.session['role'] = user.role
        return JsonResponse({'ok': True, 'user': _user_dict(user)})
    return render(request, 'register.html')


@ensure_csrf_cookie
def Dashboard(request):
    user_id = request.session.get('user_id')
    user = users.objects.filter(id=user_id).first() if user_id else None
    if not user:
        return redirect('DatabaseConectivity:login')
    return render(request, 'dashboard.html', {'user': user})


@ensure_csrf_cookie
def ForgotPassword(request):
    if request.method == 'POST':
        data = _json_body(request)
        email = (data.get('email') or '').strip().lower()
        new_password = data.get('new_password') or ''
        if not users.objects.filter(email=email).exists():
            return JsonResponse({'ok': False, 'message': 'No account found for this email.'})
        users.objects.filter(email=email).update(password=new_password)
        return JsonResponse({'ok': True})
    return render(request, 'forgot-password.html')


def Logout(request):
    request.session.flush()
    return JsonResponse({'ok': True})


def Me(request):
    user_id = request.session.get('user_id')
    user = users.objects.filter(id=user_id).first() if user_id else None
    if not user:
        return JsonResponse({'user': None})
    return JsonResponse({'user': _user_dict(user)})


def Users(request):
    if request.session.get('role') != 'admin':
        return JsonResponse({'users': []})
    user_list = list(users.objects.all().values('id', 'name', 'email', 'role'))
    return JsonResponse({'users': user_list})
