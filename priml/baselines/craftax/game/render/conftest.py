"""Keep the viewer's tests off the operator's screen.

SDL resolves its video driver during the FIRST init of any subsystem and
caches that choice for the whole process. So the request has to be made before
any test -- or any fixture, or any import -- reaches pygame; afterwards it is
too late and a window is already open.

Setting it here rather than in each test is what makes it hold: a test that
calls ``pygame.init()`` directly, as a sprite-writing fixture must, cannot
undo a choice that was already made.
"""

from __future__ import annotations

import os


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
