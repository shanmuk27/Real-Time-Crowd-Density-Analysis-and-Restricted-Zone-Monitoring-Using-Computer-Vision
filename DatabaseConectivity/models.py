from django.db import models


class Users(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    role = models.CharField(max_length=50)
<<<<<<< Updated upstream
    control = models.CharField(max_length=50)
=======
    control = models.CharField(max_length=50, default='full')
    photo = models.FileField(upload_to='photos/', blank=True, null=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'users'

    def __str__(self):
        return self.name


class Camera(models.Model):
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('danger', 'Danger'),
        ('offline', 'Offline'),
    ]
    TYPE_CHOICES = [
        ('Fixed IP Camera', 'Fixed IP Camera'),
        ('PTZ Camera', 'PTZ Camera'),
        #('360-Degree Camera', '360-Degree Camera'),
    ]

    id = models.AutoField(primary_key=True)
    camera_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    zone = models.CharField(max_length=100)
    type = models.CharField(max_length=50, choices=TYPE_CHOICES, default='Fixed IP Camera')
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

    def __str__(self):
        return f"{self.camera.camera_id}: {self.count} at {self.timestamp}"


class MonthlyCrowd(models.Model):
    id = models.AutoField(primary_key=True)
    day = models.CharField(max_length=5)
    count = models.IntegerField(default=0)
    month = models.IntegerField()
    year = models.IntegerField()

    class Meta:
        db_table = 'monthly_crowd'
        ordering = ['day']

    def __str__(self):
        return f"Day {self.day}: {self.count}"


class CameraPerson(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    verified = models.BooleanField(default=False)
    zone = models.CharField(max_length=100, blank=True, default='')
    detected_at = models.DateTimeField(auto_now_add=True)

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
>>>>>>> Stashed changes
