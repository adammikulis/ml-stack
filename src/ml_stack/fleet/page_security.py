"""What the browser is told about the page it loads: where it may fetch from and navigate to.

The page holds model replies and board messages that someone else wrote. Whatever slips
past the sanitiser still meets this policy, which lets the page read only itself: no
remote image, font, frame, stylesheet, script or connection, no form posted anywhere, no
``<base>`` rewrite and no embedding of the page in another one.
"""

CONTENT_SECURITY_POLICY = "; ".join((
    "default-src 'none'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "media-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "frame-src 'none'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
))

PAGE_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
}
