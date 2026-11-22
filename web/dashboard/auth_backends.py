"""
StudyTrade — Email or Username Authentication Backend
=====================================================

Allows users to log in with either their username (e.g. 'tushar')
or their registered email address (e.g. 'tusharpgtbiology@gmail.com').
"""

from __future__ import annotations

from typing import Any
from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from django.db.models import Q
from django.http import HttpRequest


class EmailOrUsernameBackend(ModelBackend):
    """
    Authenticate against either username or email address case-insensitively.
    """

    def authenticate(
        self,
        request: HttpRequest | None = None,
        username: str | None = None,
        password: str | None = None,
        **kwargs: Any,
    ) -> Any:
        user_model = get_user_model()
        if username is None:
            username = kwargs.get(user_model.USERNAME_FIELD)

        if not username or not password:
            return None

        try:
            # Query by either username or email
            user = user_model.objects.filter(
                Q(username__iexact=username) | Q(email__iexact=username)
            ).first()

            if user and user.check_password(password) and self.user_can_authenticate(user):
                return user
        except Exception:
            return None

        return None
