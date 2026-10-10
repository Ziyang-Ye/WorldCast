"""The recorded game's constants: the tick rate and the gravity of its engine, and its camera."""

__all__ = [
    "ASPECT_TAN_RATIO",
    "EYE_HEIGHT",
    "GRAVITY_U_PER_S2",
    "HFOV_DEGREES",
    "PITCH_MAX_DEGREES",
    "PITCH_MIN_DEGREES",
    "TICK_RATE",
]

#: Engine tick rate of the recordings, Hz.
TICK_RATE = 64.0
#: Gravity of the engine (``sv_gravity``), u / s^2.
GRAVITY_U_PER_S2 = 800.0
#: Horizontal field of view of the recordings' unscoped camera (16:9 Hor+), degrees; the camera
#: model is :mod:`worldcast.data.camera`.
HFOV_DEGREES = 106.26
#: ``tan(vfov / 2) = tan(hfov / 2) * 9 / 16`` for the cameras of the rays and of retrieval.
ASPECT_TAN_RATIO = 9.0 / 16.0
#: Camera height above the player origin (standing eye), u.
EYE_HEIGHT = 64.0
#: The engine's pitch range, degrees (pitch > 0 looks down).
PITCH_MIN_DEGREES = -89.0
PITCH_MAX_DEGREES = 89.0
