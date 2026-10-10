"""Wan2.2-TI2V-5B and the WorldCast generator built on it.

The backbone's parts: ``dit`` (the causal DiT block and the KV cache), ``attention`` (the kernels
and the training masks), ``vae`` and ``text_encoder``. ``model`` is the generator ``G_theta``, the
backbone with the conditioning of the parent package, and ``training`` its training forward.
"""
