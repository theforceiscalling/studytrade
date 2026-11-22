"""
StudyTrade — Portal Authentication & Security Middleware
=========================================================

Enforces strict authentication across the entire StudyTrade quant dashboard.
Redirects unauthenticated browser requests to `/login/` and returns HTTP 401
for unauthenticated AJAX/API calls.
"""

from __future__ import annotations

from typing import Callable
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect

EXEMPT_PREFIXES = (
    "/login/",
    "/logout/",
    "/forgot-password/",
    "/reset/",
    "/static/",
    "/admin/login/",
)


class StudyTradeAuthMiddleware:
    """
    Guarantees that no unauthorized user can access any view or API endpoint
    in the StudyTrade portal.
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        path = request.path

        # Allow exempt routes without login
        for prefix in EXEMPT_PREFIXES:
            if path.startswith(prefix):
                return self.get_response(request)

        # Check authentication
        if not request.user.is_authenticated:
            # If AJAX or API call, return JSON 401 Unauthorized
            is_ajax = (
                request.headers.get("x-requested-with") == "XMLHttpRequest"
                or path.startswith("/api/")
                or "application/json" in request.headers.get("accept", "")
            )
            if is_ajax:
                return JsonResponse(
                    {
                        "success": False,
                        "error": "Authentication required. Please log in.",
                        "login_url": f"/login/?next={path}",
                    },
                    status=401,
                )

            # Browser navigation: redirect to login page with next query param
            return redirect(f"/login/?next={path}")

        return self.get_response(request)
