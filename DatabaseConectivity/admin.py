from django.contrib import admin

from .models import users


@admin.register(users)
class UsersAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'role', 'control')
    search_fields = ('name', 'email')
    list_filter = ('role',)