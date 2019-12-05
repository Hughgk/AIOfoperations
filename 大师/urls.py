from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^main_page', views.main_page, name="main_page"),
    url(r'^$', views.index, name="index"),
    url(r'^get_result', views.get_result, name="get_result"),
    url(r'^login', views.login, name="login"),
    url(r'^register', views.register, name="register"),
]
