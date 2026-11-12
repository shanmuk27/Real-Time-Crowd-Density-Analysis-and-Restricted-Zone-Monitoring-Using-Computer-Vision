import json

import numpy as np

from django.test import TestCase
from django.utils import timezone

from .models import Camera, CameraPerson, CrowdRecord, DASHBOARD_CROWD_HISTORY_SIZE, MonthlyCrowd, Users
from .engines import VisionEngines


class CrowdRecordRetentionTests(TestCase):
    def setUp(self):
        self.camera = Camera.objects.create(
            camera_id='CAM-TEST',
            name='Retention test camera',
            location='Test location',
            zone='Test zone',
        )

    def test_only_dashboard_history_is_retained(self):
        for count in range(DASHBOARD_CROWD_HISTORY_SIZE + 3):
            CrowdRecord.record_for_dashboard(self.camera, count)

        records = list(
            CrowdRecord.objects.filter(camera=self.camera)
            .order_by('id')
            .values_list('count', flat=True)
        )

        self.assertEqual(len(records), DASHBOARD_CROWD_HISTORY_SIZE)
        self.assertEqual(records, list(range(3, DASHBOARD_CROWD_HISTORY_SIZE + 3)))

    def test_count_ingest_updates_the_current_daily_peak(self):
        self.client.post(
            '/api/crowd-record/add/',
            data=json.dumps({'camera_id': self.camera.camera_id, 'count': 7}),
            content_type='application/json',
        )
        self.client.post(
            '/api/crowd-record/add/',
            data=json.dumps({'camera_id': self.camera.camera_id, 'count': 3}),
            content_type='application/json',
        )

        current_date = timezone.localdate()
        aggregate = MonthlyCrowd.objects.get(
            day=str(current_date.day),
            month=current_date.month,
            year=current_date.year,
        )
        self.assertEqual(aggregate.count, 7)


class DashboardMetricsTests(TestCase):
    def test_metrics_report_actual_camera_feed_state(self):
        user = Users.objects.create(
            name='Dashboard user',
            email='dashboard@example.com',
            password='password',
            role='admin',
        )
        Camera.objects.create(
            camera_id='CAM-METRICS',
            name='Metrics test camera',
            location='Test location',
            zone='Test zone',
        )
        session = self.client.session
        session['user_id'] = user.id
        session.save()

        response = self.client.get('/api/dashboard-metrics/')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['camera_count'], 1)
        self.assertEqual(response.json()['running_feeds'], 0)
        self.assertEqual(response.json()['health_percent'], 0)


class UnverifiedIncidentTests(TestCase):
    def setUp(self):
        self.admin = Users.objects.create(
            name='Admin', email='admin@example.com', password='password', role='admin',
        )
        self.viewer = Users.objects.create(
            name='Viewer', email='viewer@example.com', password='password', role='analyst',
        )
        self.camera = Camera.objects.create(
            camera_id='CAM-UNKNOWN', name='Unknown test camera',
            location='Test location', zone='Restricted zone', analysis_mode='authentication',
        )

    def _login(self, user):
        session = self.client.session
        session['user_id'] = user.id
        session['role'] = user.role
        session.save()

    def test_unverified_face_is_retained_beyond_live_presence_window(self):
        person = CameraPerson.objects.create(
            camera=self.camera, name='Unknown', verified=False, zone=self.camera.zone,
            current_count=0, face_frame=b'jpeg-face',
        )
        self._login(self.viewer)

        response = self.client.get('/api/persons/')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['persons'][0]['id'], person.id)
        self.assertEqual(response.json()['persons'][0]['face_frame'], 'data:image/jpeg;base64,anBlZy1mYWNl')

    def test_only_admin_can_remove_an_unverified_incident(self):
        person = CameraPerson.objects.create(
            camera=self.camera, name='Unknown', verified=False, zone=self.camera.zone,
            face_frame=b'jpeg-face',
        )
        self._login(self.viewer)
        forbidden = self.client.post(f'/api/persons/{person.id}/remove/')
        self.assertEqual(forbidden.status_code, 403)
        self.assertTrue(CameraPerson.objects.filter(pk=person.id).exists())

        self._login(self.admin)
        removed = self.client.post(f'/api/persons/{person.id}/remove/')
        self.assertEqual(removed.status_code, 200)
        person.refresh_from_db()
        self.assertIsNotNone(person.dismissed_at)


class CameraUpdateTests(TestCase):
    def test_camera_can_be_updated_without_changing_its_id(self):
        camera = Camera.objects.create(
            camera_id='CAM-EDIT', name='Original', location='Old location', zone='Old zone', threshold=12,
        )

        response = self.client.post(
            f'/api/cameras/{camera.camera_id}/update/',
            data=json.dumps({
                'name': 'Updated camera', 'location': 'New location', 'zone': 'New zone',
                'type': 'Fixed IP Camera', 'analysis_mode': 'crowd', 'threshold': 25,
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        camera.refresh_from_db()
        self.assertEqual(camera.camera_id, 'CAM-EDIT')
        self.assertEqual(camera.name, 'Updated camera')
        self.assertEqual(camera.threshold, 25)


class VideoNoiseCleaningTests(TestCase):
    def test_cleaner_preserves_frame_shape_and_returns_a_new_cleaned_frame(self):
        # Fine-grain synthetic noise has a high residual and exercises the
        # engine's adaptive bilateral-filter path without loading a model.
        noisy = np.random.default_rng(4).integers(0, 256, (80, 80, 3), dtype=np.uint8)

        cleaned, applied = VisionEngines.clean_video_noise(noisy)

        self.assertTrue(applied)
        self.assertEqual(cleaned.shape, noisy.shape)
        self.assertEqual(cleaned.dtype, noisy.dtype)
        self.assertFalse(np.shares_memory(cleaned, noisy))

    def test_clarity_enhancer_lifts_dark_low_contrast_frames(self):
        dark = np.full((80, 80, 3), 28, dtype=np.uint8)
        dark[20:60, 20:60] = 35

        enhanced, applied = VisionEngines.enhance_frame_clarity(dark)

        self.assertTrue(applied)
        self.assertEqual(enhanced.shape, dark.shape)
        self.assertGreater(float(enhanced.mean()), float(dark.mean()))
