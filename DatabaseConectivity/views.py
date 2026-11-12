import json
import os
import base64
from collections import Counter
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.db.models import IntegerField
from django.db.models.functions import Cast
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie

from . import feeds
from .engines import engines
from .models import (
    Users, Camera, CameraEnrollment, CameraFaceCandidate, CrowdRecord, MonthlyCrowd,
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
    if user_id and Users.objects.filter(id=user_id).exists():
        return redirect('DatabaseConectivity:app')
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
def LiveCameras(request):
    if not _require_login(request):
        return redirect('DatabaseConectivity:login')
    return render(request, 'live-cameras.html')


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
        'camera_id', 'name', 'location', 'zone', 'type', 'source',
        'resolution', 'threshold', 'status', 'analysis_mode',
    ))
    result = []
    for c in cameras:
        result.append({
            'id': c['camera_id'],
            'name': c['name'],
            'location': c['location'],
            'zone': c['zone'],
            'type': c['type'],
            'source': c['source'],
            'resolution': c['resolution'],
            'threshold': c['threshold'],
            'status': c['status'],
            'analysis_mode': c['analysis_mode'],
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
    source = (data.get('source') or '').strip()
    resolution = (data.get('resolution') or '1920 × 1080').strip()
    threshold = data.get('threshold') or 50
    analysis_mode = (data.get('analysis_mode') or 'crowd').strip().lower()

    if not camera_id or not name or not location or not zone:
        return JsonResponse({'ok': False, 'message': 'Camera ID, name, location, and zone are required.'})
    if Camera.objects.filter(camera_id=camera_id).exists():
        return JsonResponse({'ok': False, 'message': f'A camera with ID {camera_id} already exists.'})
    if cam_type not in dict(Camera.TYPE_CHOICES):
        return JsonResponse({'ok': False, 'message': f'Invalid camera type "{cam_type}".'})
    if analysis_mode not in dict(Camera.ANALYSIS_MODE_CHOICES):
        return JsonResponse({'ok': False, 'message': 'Choose Crowd Count or Face Authentication.'})

    camera = Camera.objects.create(
        camera_id=camera_id, name=name, location=location,
        zone=zone, type=cam_type, source=source, resolution=resolution,
        threshold=int(threshold), analysis_mode=analysis_mode, status='active',
    )
    return JsonResponse({'ok': True, 'camera': {'id': camera.camera_id, 'analysis_mode': camera.analysis_mode}})


def ApiUpdateCamera(request, camera_id):
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'}, status=405)
    camera = Camera.objects.filter(camera_id=camera_id).first()
    if not camera:
        return JsonResponse({'ok': False, 'message': 'Camera not found.'}, status=404)
    data = _json_body(request)
    name = (data.get('name') or '').strip()
    location = (data.get('location') or '').strip()
    zone = (data.get('zone') or '').strip()
    cam_type = (data.get('type') or camera.type).strip()
    analysis_mode = (data.get('analysis_mode') or camera.analysis_mode).strip().lower()
    source = (data.get('source') or camera.source).strip()
    resolution = (data.get('resolution') or camera.resolution).strip()
    try:
        threshold = int(data.get('threshold', camera.threshold))
    except (TypeError, ValueError):
        return JsonResponse({'ok': False, 'message': 'Threshold must be a whole number.'}, status=400)
    if not name or not location or not zone:
        return JsonResponse({'ok': False, 'message': 'Camera name, location, and zone are required.'}, status=400)
    if cam_type not in dict(Camera.TYPE_CHOICES):
        return JsonResponse({'ok': False, 'message': f'Invalid camera type "{cam_type}".'}, status=400)
    if analysis_mode not in dict(Camera.ANALYSIS_MODE_CHOICES):
        return JsonResponse({'ok': False, 'message': 'Choose Crowd Count or Face Authentication.'}, status=400)
    if analysis_mode == 'crowd' and threshold < 1:
        return JsonResponse({'ok': False, 'message': 'Crowd threshold must be at least 1.'}, status=400)
    camera.name = name
    camera.location = location
    camera.zone = zone
    camera.type = cam_type
    camera.source = source
    camera.resolution = resolution
    camera.threshold = threshold if analysis_mode == 'crowd' else 999999
    camera.analysis_mode = analysis_mode
    camera.save()
    return JsonResponse({'ok': True, 'camera': {'id': camera.camera_id, 'analysis_mode': camera.analysis_mode}})


def ApiCameraFrame(request, camera_id):
    """Relay a single JPEG frame for a camera, held in memory only.

    POST stores a frame, GET returns the most recent one. Neither direction
    touches the Camera model or the database.
    """
    # 403 rather than a 200 JSON body: this URL backs an <img> src, and a
    # response of "ok: false" would render as a broken image.
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)

    if request.method == 'POST':
        frame = request.body
        if not frame:
            return JsonResponse({'ok': False, 'message': 'Empty frame body.'})
        if len(frame) > feeds.MAX_FRAME_BYTES:
            return JsonResponse({'ok': False, 'message': 'Frame too large.'})
        content_type = (request.content_type or '').lower()
        if content_type and 'image/' not in content_type:
            return JsonResponse({'ok': False, 'message': 'Expected an image body.'})
        kind = request.GET.get('kind', '')
        camera = Camera.objects.filter(camera_id=camera_id).first()
        if not camera:
            return JsonResponse({'ok': False, 'message': 'Camera not found.'}, status=404)
        enrollments = (
            list(camera.enrollments.filter(model_pack=settings.FACE_RECOGNITION_MODEL).values('name', 'embedding'))
            if camera.analysis_mode == 'authentication' else []
        )
        # Relay immediately, then let a latest-frame worker process the newest
        # image in the background. This mirrors the responsive desktop engine
        # and prevents a slow model from blocking browser uploads.
        seq = feeds.registry.publish(camera_id, frame, kind)

        def publish_engine_result(annotated_frame, crowd_count, _engine_status, auth_results):
            feeds.registry.publish_result(camera_id, annotated_frame)
            if camera.analysis_mode == 'authentication':
                current_camera = Camera.objects.filter(camera_id=camera_id).first()
                if current_camera:
                    # This is a snapshot of the latest completed inference, not
                    # a history table. It makes the dashboard reflect people
                    # actually visible in the live authentication feed.
                    # Verified identities are live presence entries. Unknown
                    # incidents intentionally persist until an administrator
                    # dismisses them, so they must not be cleared here.
                    CameraPerson.objects.filter(
                        camera=current_camera, verified=True,
                    ).update(current_count=0)
                    grouped = Counter((item['name'], item['verified']) for item in auth_results)
                    for (name, verified), seen_count in grouped.items():
                        matching_results = [
                            item for item in auth_results
                            if item['name'] == name and item['verified'] == verified
                        ]
                        best_face = max(matching_results, key=lambda item: item.get('face_quality', 0))
                        # Do not create an incident from a blurred/dark/tiny
                        # crop. A clear face frame is required for review.
                        if not verified and not best_face.get('face_frame'):
                            continue
                        person, _ = CameraPerson.objects.get_or_create(
                            camera=current_camera, name=name, verified=verified,
                            defaults={'zone': current_camera.zone, 'current_count': seen_count},
                        )
                        person.zone = current_camera.zone
                        person.current_count = seen_count
                        update_fields = ['zone', 'current_count', 'last_seen']
                        if not verified and best_face.get('face_quality', 0) >= person.face_quality:
                            person.face_frame = best_face['face_frame']
                            person.face_quality = best_face['face_quality']
                            update_fields.extend(['face_frame', 'face_quality'])
                        person.save(update_fields=update_fields)
            if crowd_count is None:
                return
            current_camera = Camera.objects.filter(camera_id=camera_id).first()
            if not current_camera:
                return
            CrowdRecord.record_for_dashboard(camera=current_camera, count=crowd_count)
            MonthlyCrowd.update_daily_peak()
            next_status = 'danger' if crowd_count >= current_camera.threshold else 'active'
            if current_camera.status != next_status:
                current_camera.status = next_status
                current_camera.save(update_fields=['status', 'updated_at'])

        engines.submit(camera_id, frame, camera.analysis_mode, enrollments, publish_engine_result)
        return JsonResponse({'ok': True, 'seq': seq, 'processing': True, 'engines': engines.status()})

    if request.method == 'GET':
        frame = feeds.registry.latest(camera_id)
        if not frame:
            return HttpResponse(status=404)
        response = HttpResponse(frame, content_type='image/jpeg')
        response['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response['Pragma'] = 'no-cache'
        return response

    return JsonResponse({'ok': False, 'message': 'GET or POST required.'})


def ApiCameraFeedStatus(request):
    """Report whether each camera's frame relay is actively running."""
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    if request.method != 'GET':
        return JsonResponse({'ok': False, 'message': 'GET required.'}, status=405)

    feeds.registry.purge()
    camera_ids = list(Camera.objects.values_list('camera_id', flat=True))
    statuses = feeds.registry.status_for(camera_ids)
    return JsonResponse({'ok': True, 'feeds': statuses})


def ApiEngineStatus(request):
    """Expose lazy model readiness without forcing a model load."""
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    return JsonResponse({'ok': True, 'engines': engines.status()})


def ApiCameraEnrollments(request, camera_id):
    """Convert uploaded enrollment photos to temporary per-camera embeddings."""
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'}, status=405)
    camera = Camera.objects.filter(camera_id=camera_id, analysis_mode='authentication').first()
    if not camera:
        return JsonResponse({'ok': False, 'message': 'Authentication camera not found.'}, status=404)
    name = (request.POST.get('name') or '').strip()
    files = request.FILES.getlist('faces')
    if not name or not files:
        return JsonResponse({'ok': False, 'message': 'Provide an identity name and at least one face image.'}, status=400)

    created = 0
    errors = []
    for face in files:
        if face.size > 8 * 1024 * 1024:
            errors.append(f'{face.name}: file is larger than 8 MB.')
            continue
        try:
            embedding = engines.enroll_face(face.read())
            CameraEnrollment.objects.create(
                camera=camera, name=name, embedding=embedding,
                model_pack=settings.FACE_RECOGNITION_MODEL,
            )
            created += 1
        except (RuntimeError, ValueError) as error:
            errors.append(f'{face.name}: {error}')
    if not created:
        return JsonResponse({'ok': False, 'message': ' '.join(errors) or 'No usable face images were supplied.'}, status=400)
    return JsonResponse({'ok': True, 'enrolled': created, 'warnings': errors})


def ApiCameraFaceCandidates(request, camera_id):
    """Detect faces from a browser-selected folder before an operator names them."""
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'}, status=405)
    camera = Camera.objects.filter(camera_id=camera_id, analysis_mode='authentication').first()
    if not camera:
        return JsonResponse({'ok': False, 'message': 'Authentication camera not found.'}, status=404)
    files = request.FILES.getlist('faces')
    if not files:
        return JsonResponse({'ok': False, 'message': 'Select a folder containing face images.'}, status=400)

    CameraFaceCandidate.objects.filter(camera=camera).delete()
    candidates = []
    errors = []
    for face in files[:50]:
        if face.size > 8 * 1024 * 1024:
            errors.append(f'{face.name}: larger than 8 MB.')
            continue
        try:
            for embedding, thumbnail in engines.extract_face_candidates(face.read()):
                candidate = CameraFaceCandidate.objects.create(
                    camera=camera, embedding=embedding, thumbnail=thumbnail,
                )
                candidates.append({
                    'id': candidate.id,
                    'thumbnail': 'data:image/jpeg;base64,' + base64.b64encode(thumbnail).decode('ascii'),
                })
                if len(candidates) >= 30:
                    break
        except (RuntimeError, ValueError) as error:
            errors.append(f'{face.name}: {error}')
        if len(candidates) >= 30:
            break
    if not candidates:
        return JsonResponse({'ok': False, 'message': 'No usable faces were detected. ' + ' '.join(errors)}, status=400)
    return JsonResponse({'ok': True, 'candidates': candidates, 'warnings': errors})


def ApiConfirmCameraFaceCandidates(request, camera_id):
    """Turn named temporary candidates into enrolled authentication identities."""
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'}, status=405)
    camera = Camera.objects.filter(camera_id=camera_id, analysis_mode='authentication').first()
    entries = _json_body(request).get('candidates') or []
    if not camera or not entries:
        return JsonResponse({'ok': False, 'message': 'Provide at least one named detected face.'}, status=400)
    candidates = {candidate.id: candidate for candidate in CameraFaceCandidate.objects.filter(camera=camera)}
    created = 0
    for entry in entries:
        try:
            candidate = candidates.get(int(entry.get('id')))
        except (TypeError, ValueError):
            candidate = None
        name = (entry.get('name') or '').strip() if isinstance(entry, dict) else ''
        if candidate and name:
            CameraEnrollment.objects.create(
                camera=camera, name=name, embedding=candidate.embedding,
                model_pack=settings.FACE_RECOGNITION_MODEL,
            )
            created += 1
    CameraFaceCandidate.objects.filter(camera=camera).delete()
    if not created:
        return JsonResponse({'ok': False, 'message': 'Name at least one detected face before saving.'}, status=400)
    return JsonResponse({'ok': True, 'enrolled': created})


def _format_file_size(size_bytes):
    """Format a byte count for the small dashboard summary card."""
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    size = float(max(0, size_bytes))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f'{size:.1f} {unit}' if unit != 'B' else f'{int(size)} {unit}'
        size /= 1024


def ApiDashboardMetrics(request):
    """Return current, displayable operational values for the dashboard."""
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    if request.method != 'GET':
        return JsonResponse({'ok': False, 'message': 'GET required.'}, status=405)

    database_name = settings.DATABASES['default'].get('NAME', '')
    database_bytes = (
        os.path.getsize(database_name)
        if database_name and os.path.isfile(database_name)
        else 0
    )

    feeds.registry.purge()
    camera_ids = list(Camera.objects.values_list('camera_id', flat=True))
    feed_statuses = feeds.registry.status_for(camera_ids)
    running_feeds = sum(status['running'] for status in feed_statuses.values())
    camera_count = len(camera_ids)
    health_percent = round((running_feeds / camera_count) * 100) if camera_count else None

    return JsonResponse({
        'ok': True,
        'database_storage': _format_file_size(database_bytes),
        'database_storage_bytes': database_bytes,
        'camera_count': camera_count,
        'running_feeds': running_feeds,
        'health_percent': health_percent,
    })


def ApiRemoveCamera(request, camera_id):
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'})
    deleted, _ = Camera.objects.filter(camera_id=camera_id).delete()
    if deleted:
        return JsonResponse({'ok': True})
    return JsonResponse({'ok': False, 'message': 'Camera not found.'})


def ApiLiveCrowd(request):
    latest_records = []
    # Authentication cameras report identity presence in Identified Area; they
    # must not be represented as zero-person crowd cameras in this summary.
    crowd_cameras = Camera.objects.filter(analysis_mode='crowd')
    for cam in crowd_cameras:
        rec = CrowdRecord.objects.filter(camera=cam).order_by('-timestamp').first()
        if rec:
            latest_records.append(rec)

    latest_records.sort(key=lambda r: r.timestamp)
    data = []
    for r in latest_records:
        data.append({
            'time': r.timestamp.strftime('%H:%M'),
            'count': r.count,
            'camera': r.camera.camera_id,
        })

    per_camera = {r.camera.camera_id: r.count for r in latest_records}
    history = {}
    for cam in crowd_cameras:
        series = list(
            CrowdRecord.objects.filter(camera=cam)
            .order_by('-timestamp')
            .values_list('count', flat=True)[:12]
        )
        series.reverse()
        if series:
            history[cam.camera_id] = series

    # Build a chronological total-crowd series from the retained live samples.
    # Each point updates one camera's current value, then records the new total.
    # This produces a real time graph even when there is only one camera.
    events = list(
        CrowdRecord.objects.filter(camera__analysis_mode='crowd')
        .select_related('camera')
        .order_by('timestamp', 'id')
        .values('camera__camera_id', 'count', 'timestamp')
    )
    running_counts = {}
    timeline = []
    for event in events:
        running_counts[event['camera__camera_id']] = event['count']
        timeline.append({
            'time': event['timestamp'].strftime('%H:%M:%S'),
            'count': sum(running_counts.values()),
        })
    timeline = timeline[-12:]

    counts = [d['count'] for d in data]
    total = sum(counts)
    peak = max(counts, default=0)
    avg = round(total / len(counts)) if counts else 0

    return JsonResponse({
        'live_crowd': data,
        'total': total,
        'peak': peak,
        'avg': avg,
        'camera_count': crowd_cameras.count(),
        'per_camera': per_camera,
        'history': history,
        'timeline': timeline,
    })


@csrf_exempt
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
    # Raw detector output is sampled only into the dashboard's fixed history
    # window. This prevents a long-running feed from growing the database
    # without bound while preserving every point the homepage can display.
    CrowdRecord.record_for_dashboard(camera=cam, count=int(count or 0))
    MonthlyCrowd.update_daily_peak()
    if count and int(count) >= cam.threshold:
        cam.status = 'danger'
    else:
        cam.status = 'active'
    cam.save()
    return JsonResponse({'ok': True})


def ApiMonthlyCrowd(request):
    now = timezone.localdate()
    month = int(request.GET.get('month', now.month))
    year = int(request.GET.get('year', now.year))
    records = list(
        MonthlyCrowd.objects.filter(month=month, year=year)
        .annotate(day_number=Cast('day', IntegerField()))
        .order_by('day_number')
        .values('day', 'count')
    )
    return JsonResponse({'monthly_crowd': records, 'month': month, 'year': year})


def ApiCameraPersons(request):
    if not _require_login(request):
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    cutoff = timezone.now() - timedelta(seconds=15)
    persons = list(
        CameraPerson.objects.filter(
            camera__isnull=False, dismissed_at__isnull=True,
        ).filter(
            # Verified people describe current presence; unverified incidents
            # stay visible until an administrator explicitly removes them.
            models.Q(verified=False) | models.Q(current_count__gt=0, last_seen__gte=cutoff),
        ).values('id', 'name', 'verified', 'zone', 'camera__camera_id', 'current_count', 'face_frame')
    )
    for person in persons:
        person['camera_id'] = person.pop('camera__camera_id')
        frame = person.pop('face_frame')
        person['face_frame'] = (
            'data:image/jpeg;base64,' + base64.b64encode(frame).decode('ascii')
            if frame else None
        )
    return JsonResponse({'persons': persons})


def ApiRemoveUnverifiedPerson(request, person_id):
    """Dismiss a retained unknown-person incident from the dashboard."""
    user = _require_login(request)
    if not user:
        return JsonResponse({'ok': False, 'message': 'Not authenticated.'}, status=403)
    if user.role.strip().lower() != 'admin':
        return JsonResponse({'ok': False, 'message': 'Only administrators can remove unverified incidents.'}, status=403)
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'message': 'POST required.'}, status=405)
    person = CameraPerson.objects.filter(
        pk=person_id, verified=False, dismissed_at__isnull=True,
    ).first()
    if not person:
        return JsonResponse({'ok': False, 'message': 'Unverified incident not found.'}, status=404)
    # Keep the incident as an audit record but remove it permanently from the
    # active dashboard list. The incoming camera stream must not recreate it.
    person.dismissed_at = timezone.now()
    person.save(update_fields=['dismissed_at'])
    return JsonResponse({'ok': True})


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
