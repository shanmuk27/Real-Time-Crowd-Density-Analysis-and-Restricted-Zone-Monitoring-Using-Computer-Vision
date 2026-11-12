from django.db import models, transaction
from django.utils import timezone


# The dashboard renders one current reading and a 12-point sparkline for each
# camera. Retaining more raw detector readings does not add anything to that
# view, so keep the database bounded to exactly that useful history window.
DASHBOARD_CROWD_HISTORY_SIZE = 12


class Users(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    role = models.CharField(max_length=50)
    control = models.CharField(max_length=50, default='full')
    photo = models.FileField(upload_to='photos/', blank=True, null=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'users'

    def __str__(self):
        return self.name


class Camera(models.Model):
    ANALYSIS_MODE_CHOICES = [
        ('crowd', 'Crowd Count'),
        ('authentication', 'Face Authentication'),
    ]
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('danger', 'Danger'),
        ('offline', 'Offline'),
    ]
    TYPE_CHOICES = [
        ('Fixed IP Camera', 'Fixed IP Camera'),
        ('Video File', 'Video File'),
        ('Web Cam', 'Web Cam'),
    ]

    id = models.AutoField(primary_key=True)
    camera_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    zone = models.CharField(max_length=100)
    type = models.CharField(max_length=50, choices=TYPE_CHOICES, default='Fixed IP Camera')
    source = models.CharField(max_length=255, blank=True, default='')
    analysis_mode = models.CharField(max_length=20, choices=ANALYSIS_MODE_CHOICES, default='crowd')
    resolution = models.CharField(max_length=20, default='1920 × 1080')
    threshold = models.IntegerField(default=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cameras'

    def __str__(self):
        return f"{self.camera_id} - {self.name}"


class CrowdRecord(models.Model):
    id = models.AutoField(primary_key=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='crowd_records')
    count = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'crowd_records'
        ordering = ['-timestamp']
        indexes = [
            models.Index(
                fields=['camera', '-timestamp', '-id'],
                name='crowd_record_dashboard_idx',
            ),
        ]

    @classmethod
    def record_for_dashboard(cls, camera, count):
        """Store a count and retain only the history the dashboard displays.

        Locking the camera makes simultaneous ingest requests for the same
        camera prune safely instead of allowing the retained history to grow.
        """
        with transaction.atomic():
            locked_camera = Camera.objects.select_for_update().get(pk=camera.pk)
            record = cls.objects.create(camera=locked_camera, count=count)
            stale_ids = list(
                cls.objects.filter(camera=locked_camera)
                .order_by('-timestamp', '-id')
                .values_list('id', flat=True)[DASHBOARD_CROWD_HISTORY_SIZE:]
            )
            if stale_ids:
                cls.objects.filter(id__in=stale_ids).delete()
        return record

    def __str__(self):
        return f"{self.camera.camera_id}: {self.count} at {self.timestamp}"


class CameraEnrollment(models.Model):
    """An in-database face embedding used only by one authentication camera.

    Source photos are converted to embeddings during enrollment and are not
    retained. The records cascade with their camera, so removing a camera also
    removes every enrolled identity for it.
    """
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='enrollments')
    name = models.CharField(max_length=100)
    embedding = models.JSONField()
    model_pack = models.CharField(max_length=40, default='buffalo_l')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'camera_enrollments'

    def __str__(self):
        return f"{self.camera.camera_id}: {self.name}"


class CameraFaceCandidate(models.Model):
    """Temporary detected face awaiting an operator-supplied identity name."""
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='face_candidates')
    embedding = models.JSONField()
    thumbnail = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'camera_face_candidates'


class MonthlyCrowd(models.Model):
    id = models.AutoField(primary_key=True)
    day = models.CharField(max_length=5)
    count = models.IntegerField(default=0)
    month = models.IntegerField()
    year = models.IntegerField()

    class Meta:
        db_table = 'monthly_crowd'
        ordering = ['day']

    @classmethod
    def update_daily_peak(cls):
        """Record the day's peak live crowd total for the monthly chart.

        A single daily aggregate is useful to the homepage; individual frame
        counts are deliberately not copied into this table.
        """
        current_date = timezone.localdate()
        live_total = 0
        for camera in Camera.objects.all():
            latest = (
                CrowdRecord.objects.filter(camera=camera)
                .order_by('-timestamp', '-id')
                .values_list('count', flat=True)
                .first()
            )
            live_total += latest or 0

        daily_record, created = cls.objects.get_or_create(
            day=str(current_date.day),
            month=current_date.month,
            year=current_date.year,
            defaults={'count': live_total},
        )
        if not created and live_total > daily_record.count:
            daily_record.count = live_total
            daily_record.save(update_fields=['count'])
        return daily_record

    def __str__(self):
        return f"Day {self.day}: {self.count}"


class CameraPerson(models.Model):
    id = models.AutoField(primary_key=True)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='identified_persons', null=True, blank=True)
    name = models.CharField(max_length=100)
    verified = models.BooleanField(default=False)
    zone = models.CharField(max_length=100, blank=True, default='')
    detected_at = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)
    current_count = models.PositiveIntegerField(default=1)
    # Unknown-person incidents are retained for administrator review. This is
    # a small JPEG crop of the clearest detected face, never the full frame.
    face_frame = models.BinaryField(blank=True, null=True)
    face_quality = models.FloatField(default=0)
    dismissed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'camera_persons'

    def __str__(self):
        return f"{self.name} ({'verified' if self.verified else 'unverified'})"


class Notification(models.Model):
    id = models.AutoField(primary_key=True)
    icon = models.CharField(max_length=10, default='🔔')
    message = models.CharField(max_length=500)
    unread = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'notifications'
        ordering = ['-created_at']

    def __str__(self):
        return self.message
