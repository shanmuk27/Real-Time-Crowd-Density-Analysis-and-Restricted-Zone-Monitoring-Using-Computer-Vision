from django.core.management.base import BaseCommand

from DatabaseConectivity.models import (
    Users, Camera, CrowdRecord, MonthlyCrowd,
    CameraPerson, Notification,
)


class Command(BaseCommand):
    help = 'Seed database with initial demo data'

    def handle(self, *args, **options):
        self.stdout.write('Seeding database...')

        # Users
        users_data = [
            ('John Doe', 'john@example.com', 'admin', 'full'),
            ('Jane Smith', 'jane@example.com', 'analyst', 'full'),
            ('Alex Wang', 'alex@example.com', 'developer', 'full'),
            ('Maria Garcia', 'maria@example.com', 'client', 'full'),
            ('Sam Johnson', 'sam@example.com', 'analyst', 'full'),
            ('Emma Wilson', 'emma@example.com', 'developer', 'full'),
            ('David Brown', 'david@example.com', 'client', 'full'),
            ('Sarah Miller', 'sarah@example.com', 'analyst', 'full'),
        ]
        for name, email, role, control in users_data:
            Users.objects.get_or_create(
                email=email,
                defaults={'name': name, 'password': 'password123', 'role': role, 'control': control},
            )
        self.stdout.write(self.style.SUCCESS(f'Created {len(users_data)} users'))

        # Cameras
        cameras_data = [
            ('CAM-001', 'Entrance Gate A', 'Main Building - South Entrance', 'Public Zone', 'Fixed IP Camera', '1920 × 1080', 80, 'active'),
            ('CAM-002', 'Restricted Zone B', 'Server Room - Floor 2', 'Restricted Zone', 'PTZ Camera', '2560 × 1440', 30, 'danger'),
            ('CAM-003', 'Parking Area', 'North Parking Lot', 'Public Zone', 'Fixed IP Camera', '1280 × 720', 60, 'active'),
            ('CAM-004', 'Main Hall', 'Auditorium - Floor 1', 'Public Zone', '360-Degree Camera', '3840 × 2160', 100, 'active'),
            ('CAM-005', 'Corridor C', 'Office Wing - Floor 3', 'Staff Zone', 'Fixed IP Camera', '1920 × 1080', 40, 'active'),
            ('CAM-006', 'Stairwell D', 'Emergency Exit - West', 'Restricted Zone', 'Fixed IP Camera', '1280 × 720', 25, 'offline'),
        ]
        for cam_id, name, loc, zone, cam_type, res, thresh, status in cameras_data:
            Camera.objects.get_or_create(
                camera_id=cam_id,
                defaults={
                    'name': name, 'location': loc, 'zone': zone,
                    'type': cam_type, 'resolution': res,
                    'threshold': thresh, 'status': status,
                },
            )
        self.stdout.write(self.style.SUCCESS(f'Created {len(cameras_data)} cameras'))

        # Monthly crowd data
        monthly_data = [
            ('1', 68), ('2', 91), ('3', 55), ('4', 120), ('5', 44),
            ('6', 132), ('7', 87), ('8', 63), ('9', 108), ('10', 75),
            ('11', 96), ('12', 58), ('13', 140), ('14', 39), ('15', 112),
            ('16', 70), ('17', 84), ('18', 101), ('19', 66), ('20', 124),
            ('21', 53), ('22', 89), ('23', 145), ('24', 61), ('25', 98),
            ('26', 72), ('27', 118), ('28', 47), ('29', 105), ('30', 83),
        ]
        import datetime
        now = datetime.datetime.now()
        MonthlyCrowd.objects.filter(month=now.month, year=now.year).delete()
        for day, count in monthly_data:
            MonthlyCrowd.objects.create(day=day, count=count, month=now.month, year=now.year)
        self.stdout.write(self.style.SUCCESS(f'Created {len(monthly_data)} monthly crowd records'))

        # Camera persons
        persons_data = [
            ('Unknown', False, 'Entrance Gate A'),
            ('Unknown', False, 'Restricted Zone B'),
            ('Unknown', False, 'Parking Area'),
            ('John Doe', True, ''),
            ('Jane Smith', True, ''),
            ('Alex Wang', True, ''),
            ('Maria Garcia', True, ''),
            ('David Johnson', True, ''),
            ('Priya Patel', True, ''),
            ('Michael Brown', True, ''),
            ('Sofia Rossi', True, ''),
            ('Liam Chen', True, ''),
            ('Emma Wilson', True, ''),
            ('Noah Kim', True, ''),
        ]
        CameraPerson.objects.all().delete()
        for name, verified, zone in persons_data:
            CameraPerson.objects.create(name=name, verified=verified, zone=zone)
        self.stdout.write(self.style.SUCCESS(f'Created {len(persons_data)} camera persons'))

        # Notifications
        notifications_data = [
            ('📷', 'Camera CAM-004 went offline', True),
            ('⚠️', 'High crowd density detected in Zone A', True),
            ('✅', 'System health check completed successfully', True),
            ('🔐', 'New admin login detected', False),
            ('📊', 'Daily crowd report is ready', False),
            ('🔔', 'Camera firmware update available', False),
        ]
        Notification.objects.all().delete()
        for icon, message, unread in notifications_data:
            Notification.objects.create(icon=icon, message=message, unread=unread)
        self.stdout.write(self.style.SUCCESS(f'Created {len(notifications_data)} notifications'))

        self.stdout.write(self.style.SUCCESS('Database seeded successfully!'))
