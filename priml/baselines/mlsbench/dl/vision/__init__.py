"""The scaffold the MLS-Bench DL tasks share: data, model, step and recipe.

``exp000`` is the recipe of the ``cv-classification-loss`` script at its
``resnet56-cifar100`` setting. Each task package forks it under its own study
name and makes the task's editable function a slot, so a new idea is a fork
rather than an edit of a copy.
"""
