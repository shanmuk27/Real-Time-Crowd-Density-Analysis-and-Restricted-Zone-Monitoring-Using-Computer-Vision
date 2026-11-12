from django.db import migrations


DEMO_MESSAGES = [
    'Camera CAM-004 went offline', 'High crowd density detected in Zone A',
    'System health check completed successfully', 'New admin login detected',
    'Daily crowd report is ready', 'Camera firmware update available',
]


def remove_demo_notifications(apps, schema_editor):
    apps.get_model('DatabaseConectivity', 'Notification').objects.filter(message__in=DEMO_MESSAGES).delete()


class Migration(migrations.Migration):
    dependencies = [('DatabaseConectivity', '0012_cameraperson_dismissed_at')]
    operations = [migrations.RunPython(remove_demo_notifications, migrations.RunPython.noop)]
