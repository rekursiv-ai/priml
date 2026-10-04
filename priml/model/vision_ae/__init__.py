"""Pretrained vision autoencoders behind one encode/decode contract.

Each module wraps one published checkpoint family -- INVAE, RAE, VTP -- so a
consumer holds an :class:`~priml.model.vision_ae.custom_types.Autoencoder` and
never branches on which one it has. Three transforms stay apart: the
autoencoder maps pixels to its NATIVE latent, a latent normalizer maps that to
the space a diffusion model trains in, and a storage codec (owned by the
dataset that materializes a corpus) maps it to bytes on disk.
"""
