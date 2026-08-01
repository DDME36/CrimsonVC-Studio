"""
Web package for the Ultimate RVC project.

This package contains modules which define the web application of the
Ultimate RVC project.
"""

from __future__ import annotations

import asyncio
import os

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
