import json
<<<<<<< Updated upstream
=======
import os
from datetime import datetime
>>>>>>> Stashed changes

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import (
    Users, Camera, CrowdRecord, MonthlyCrowd,
    CameraPerson, Notification,
)


def _json_body(request):
    try:
        return json.loads(request.body or b'{}')
    except (ValueError, TypeError):
        return {}


def _user_dict(user):
    return {'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role}


def _require_login(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return None
    return Users.objects.filter(id=user_id).first()


@ensure_csrf_cookie
def Home(request):
    user_id = request.session.get('user_id')
<<<<<<< Updated upstream
    if user_id and users.objects.filter(id=user_id).exists():
        return redirect('DatabaseConectivity:dashboard')
=======
    if user_id and Users.objects.filter(id=user_id).exists():
        return redirect('DatabaseConectivity:app')
>>>>>>> Stashed changes
    return redirect('DatabaseConectivity:login')


def Login(request):
    if request.method == 'POST':
        data = _json_body(request)
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        try:
            user = Users.objects.get(email=email)
        except Users.DoesNotExist:
            return JsonResponse({'ok': False, 'message': 'No account found for this email. Please register first.'})
        if user.password != password:
            return JsonResponse({'ok': False, 'message': 'Incorrect password. Please try again.'})
        request.session.flush()
        request.session['user_id'] = user.id
        request.session['role'] = user.role
        return JsonResponse({'ok': True, 'user': _user_dict(user)})
    return render(request, 'login.html')


@ensure_csrf_cookie
<<<<<<< Updated upstream
=======
def App(request):
    user = _require_login(request)
    if not user:
        return redirect('DatabaseConectivity:login')
    return render(request, 'app.html', {'user': user})


@ensure_csrf_cookie
def CameraManagement(request):
    if not _require_login(request):
        return redirect('DatabaseConectivity:login')
    return render(request, 'camera.html')


@ensure_csrf_cookie
def Profile(request):
    user = _require_login(request)
    if not user:
        return redirect('DatabaseConectivity:login')
    return render(request, 'profile.html', {'user': user})


def UploadPhoto(request):
    user = _require_login(request)
    if not user:
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'})
    photo = request.FILES.get('photo')
    if not photo:
        return JsonResponse({'ok': False, 'message': 'No photo provided.'})
    ext = os.path.splitext(photo.name)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
        return JsonResponse({'ok': False, 'message': 'Unsupported file type. Use JPG, PNG, GIF or WEBP.'})
    if user.photo:
        default_storage.delete(user.photo.name)
    user.photo.save('user_{0}{1}'.format(user.id, ext), photo, save=True)
    return JsonResponse({'ok': True, 'photo': user.photo.url})


def UpdateProfile(request):
    user = _require_login(request)
    if not user:
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'})
    data = _json_body(request)
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    if not name or not email:
        return JsonResponse({'ok': False, 'message': 'Both fields are required.'})
    existing = Users.objects.filter(email=email).exclude(id=user.id).first()
    if existing:
        return JsonResponse({'ok': False, 'message': 'An account with this email already exists.'})
    user.name = name
    user.email = email
    user.save()
    return JsonResponse({'ok': True, 'user': _user_dict(user)})


@ensure_csrf_cookie
>>>>>>> Stashed changes
def Register(request):
    if request.method == 'POST':
        data = _json_body(request)
        name = (data.get('name') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        role = data.get('role') or ''
        if not name or not email or not password or not role:
            return JsonResponse({'ok': False, 'message': 'All fields are required.'})
        if Users.objects.filter(email=email).exists():
            return JsonResponse({'ok': False, 'message': 'An account with this email already exists. Try logging in instead.'})
        user = Users.objects.create(name=name, email=email, password=password, role=role, control='full')
        request.session.flush()
        request.session['user_id'] = user.id
        request.session['role'] = user.role
        return JsonResponse({'ok': True, 'user': _user_dict(user)})
    return render(request, 'register.html')


@ensure_csrf_cookie
def Dashboard(request):
    user = _require_login(request)
    if not user:
        return redirect('DatabaseConectivity:login')
    return render(request, 'dashboard.html', {'user': user})


@ensure_csrf_cookie
def ForgotPassword(request):
    if request.method == 'POST':
        data = _json_body(request)
        email = (data.get('email') or '').strip().lower()
        new_password = data.get('new_password') or ''
        if not Users.objects.filter(email=email).exists():
            return JsonResponse({'ok': False, 'message': 'No account found for this email.'})
        Users.objects.filter(email=email).update(password=new_password)
        return JsonResponse({'ok': True})
    return render(request, 'forgot-password.html')


def Logout(request):
    request.session.flush()
    return JsonResponse({'ok': True})


def Me(request):
    user = _require_login(request)
    if not user:
        return JsonResponse({'user': None})
    return JsonResponse({'user': _user_dict(user)})


def UsersList(request):
    if request.session.get('role') != 'admin':
        return JsonResponse({'users': []})
    user_list = list(Users.objects.all().values('id', 'name', 'email', 'role'))
    return JsonResponse({'users': user_list})


# ---------- API Views ----------

def ApiCameras(request):
    cameras = list(Camera.objects.all().values(
        'camera_id', 'name', 'location', 'zone', 'type',
        'resolution', 'threshold', 'status',
    ))
    result = []
    for c in cameras:
        result.append({
            'id': c['camera_id'],
            'name': c['name'],
            'location': c['location'],
            'zone': c['zone'],
            'type': c['type'],
            'resolution': c['resolution'],
            'threshold': c['threshold'],
            'status': c['status'],
        })
    return JsonResponse({'cameras': result})


def ApiAddCamera(request):
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'})
    data = _json_body(request)
    camera_id = (data.get('id') or '').strip()
    name = (data.get('name') or '').strip()
    location = (data.get('location') or '').strip()
    zone = (data.get('zone') or '').strip()
    cam_type = (data.get('type') or 'Fixed IP Camera').strip()
    resolution = (data.get('resolution') or '1920 × 1080').strip()
    threshold = data.get('threshold') or 50

    if not camera_id or not name or not location or not zone:
        return JsonResponse({'ok': False, 'message': 'Camera ID, name, location, and zone are required.'})
    if Camera.objects.filter(camera_id=camera_id).exists():
        return JsonResponse({'ok': False, 'message': f'A camera with ID {camera_id} already exists.'})

    Camera.objects.create(
        camera_id=camera_id, name=name, location=location,
        zone=zone, type=cam_type, resolution=resolution,
        threshold=int(threshold), status='active',
    )
    return JsonResponse({'ok': True})


def ApiRemoveCamera(request, camera_id):
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'})
    deleted, _ = Camera.objects.filter(camera_id=camera_id).delete()
    if deleted:
        return JsonResponse({'ok': True})
    return JsonResponse({'ok': False, 'message': 'Camera not found.'})


def ApiLiveCrowd(request):
    records = CrowdRecord.objects.select_related('camera').order_by('timestamp')[:50]
    data = []
    for r in records:
        data.append({
            'time': r.timestamp.strftime('%H:%M'),
            'count': r.count,
            'camera': r.camera.camera_id,
        })
    total = sum(d['count'] for d in data)
    peak = max((d['count'] for d in data), default=0)
    avg = round(total / len(data)) if data else 0
    return JsonResponse({
        'live_crowd': data,
        'total': total,
        'peak': peak,
        'avg': avg,
        'camera_count': Camera.objects.count(),
    })


def ApiAddCrowdRecord(request):
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'})
    data = _json_body(request)
    camera_id = (data.get('camera_id') or '').strip()
    count = data.get('count')
    try:
        cam = Camera.objects.get(camera_id=camera_id)
    except Camera.DoesNotExist:
        return JsonResponse({'ok': False, 'message': 'Camera not found.'})
    CrowdRecord.objects.create(camera=cam, count=int(count or 0))
    if count and int(count) >= cam.threshold:
        cam.status = 'danger'
    else:
        cam.status = 'active'
    cam.save()
    return JsonResponse({'ok': True})


def ApiMonthlyCrowd(request):
    now = datetime.now()
    month = int(request.GET.get('month', now.month))
    year = int(request.GET.get('year', now.year))
    records = list(MonthlyCrowd.objects.filter(month=month, year=year).values('day', 'count'))
    return JsonResponse({'monthly_crowd': records})


def ApiCameraPersons(request):
    persons = list(CameraPerson.objects.all().values('id', 'name', 'verified', 'zone'))
    return JsonResponse({'persons': persons})


def ApiNotifications(request):
    notifs = list(Notification.objects.all().values('icon', 'message', 'unread', 'created_at'))
    result = []
    for n in notifs:
        result.append({
            'icon': n['icon'],
            'message': n['message'],
            'unread': n['unread'],
            'time': n['created_at'].strftime('%Y-%m-%d %H:%M'),
        })
    return JsonResponse({'notifications': result})


def ApiTeamMembers(request):
    members = list(Users.objects.all().values('id', 'name', 'email', 'role', 'is_active'))
    result = []
    for m in members:
        result.append({
            'id': m['id'],
            'name': m['name'],
            'email': m['email'],
            'role': m['role'],
            'active': m['is_active'],
        })
    return JsonResponse({'members': result})
