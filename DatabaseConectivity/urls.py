from django.urls import path

from . import views

app_name = 'DatabaseConectivity'

urlpatterns = [
    path('', views.Home, name='home'),
    path('login/', views.Login, name='login'),
    path('register/', views.Register, name='register'),
    path('dashboard/', views.Dashboard, name='dashboard'),
    path('forgot-password/', views.ForgotPassword, name='forgot-password'),
    path('logout/', views.Logout, name='logout'),
    path('me/', views.Me, name='me'),
    path('users/', views.UsersList, name='users'),
    path('app/', views.App, name='app'),
    path('camera-management/', views.CameraManagement, name='camera-management'),
    path('live-cameras/', views.LiveCameras, name='live-cameras'),
    path('profile/', views.Profile, name='profile'),
    path('profile/photo/', views.UploadPhoto, name='upload-photo'),
    path('profile/edit/', views.UpdateProfile, name='edit-profile'),
    # API endpoints
    path('api/cameras/', views.ApiCameras, name='api-cameras'),
    path('api/cameras/add/', views.ApiAddCamera, name='api-add-camera'),
    path('api/cameras/<str:camera_id>/update/', views.ApiUpdateCamera, name='api-update-camera'),
    path('api/cameras/<str:camera_id>/remove/', views.ApiRemoveCamera, name='api-remove-camera'),
    path('api/cameras/<str:camera_id>/frame/', views.ApiCameraFrame, name='api-camera-frame'),
    path('api/cameras/<str:camera_id>/enrollments/', views.ApiCameraEnrollments, name='api-camera-enrollments'),
    path('api/cameras/<str:camera_id>/face-candidates/', views.ApiCameraFaceCandidates, name='api-camera-face-candidates'),
    path('api/cameras/<str:camera_id>/face-candidates/confirm/', views.ApiConfirmCameraFaceCandidates, name='api-confirm-camera-face-candidates'),
    path('api/camera-feeds/status/', views.ApiCameraFeedStatus, name='api-camera-feed-status'),
    path('api/engines/status/', views.ApiEngineStatus, name='api-engine-status'),
    path('api/dashboard-metrics/', views.ApiDashboardMetrics, name='api-dashboard-metrics'),
    path('api/live-crowd/', views.ApiLiveCrowd, name='api-live-crowd'),
    path('api/crowd-record/add/', views.ApiAddCrowdRecord, name='api-add-crowd-record'),
    path('api/monthly-crowd/', views.ApiMonthlyCrowd, name='api-monthly-crowd'),
    path('api/persons/', views.ApiCameraPersons, name='api-persons'),
    path('api/persons/<int:person_id>/remove/', views.ApiRemoveUnverifiedPerson, name='api-remove-unverified-person'),
    path('api/notifications/', views.ApiNotifications, name='api-notifications'),
    path('api/team/', views.ApiTeamMembers, name='api-team'),
]
