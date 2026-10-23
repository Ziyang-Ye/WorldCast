"""The recorded game's constants: the tick rate and the gravity of its engine, its camera and its
players' bodies."""

__all__ = [
    "ASPECT_TAN_RATIO",
    "BODY_HEIGHT",
    "BODY_HEIGHT_CROUCHED",
    "CROUCHED_EYE_HEIGHT",
    "EYE_HEIGHT",
    "GRAVITY_U_PER_S2",
    "HFOV_DEGREES",
    "PITCH_MAX_DEGREES",
    "PITCH_MIN_DEGREES",
    "PLAYER_WIDTH",
    "TICK_RATE",
    "VFOV_DEGREES",
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
#: Vertical field of view of the unscoped camera, degrees, with which the player state field, the
#: visibility labels and the boxes of the entity criterion project: the rounded 73.74 they were
#: made with, not the ``tan(hfov / 2) * 9 / 16`` of the rays and of retrieval
#: (:data:`ASPECT_TAN_RATIO`).
VFOV_DEGREES = 73.74
#: Camera height above the player origin (standing eye), u.
EYE_HEIGHT = 64.0
#: Camera height above the player origin of a crouched player, u.
CROUCHED_EYE_HEIGHT = 46.0
#: Height of a player's body above its origin (the feet), standing and crouched, u.
BODY_HEIGHT = 72.0
BODY_HEIGHT_CROUCHED = 54.0
#: Width of a player's body, u (the engine's player hull).
PLAYER_WIDTH = 32.0
#: The engine's pitch range, degrees (pitch > 0 looks down).
PITCH_MIN_DEGREES = -89.0
PITCH_MAX_DEGREES = 89.0
