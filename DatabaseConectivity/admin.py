from django.contrib import admin

from .models import (
    Users, Camera, CrowdRecord, MonthlyCrowd,
    CameraPerson, Notification,
)


@admin.register(Users)
class UsersAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'role', 'control', 'is_active')
    search_fields = ('name', 'email')
    list_filter = ('role', 'is_active')


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ('camera_id', 'name', 'location', 'zone', 'type', 'threshold', 'status')
    search_fields = ('camera_id', 'name', 'location')
    list_filter = ('status', 'zone', 'type')


@admin.register(CrowdRecord)
class CrowdRecordAdmin(admin.ModelAdmin):
    list_display = ('camera', 'count', 'timestamp')
    list_filter = ('camera',)


@admin.register(MonthlyCrowd)
class MonthlyCrowdAdmin(admin.ModelAdmin):
    list_display = ('day', 'count', 'month', 'year')


@admin.register(CameraPerson)
class CameraPersonAdmin(admin.ModelAdmin):
    list_display = ('name', 'verified', 'zone', 'detected_at', 'last_seen', 'dismissed_at', 'face_preview')
    list_filter = ('verified',)

    @admin.display(description='Stored face')
    def face_preview(self, person):
        return 'Yes' if person.face_frame else '—'


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('icon', 'message', 'unread', 'created_at')
    list_filter = ('unread',)
