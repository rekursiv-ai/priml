"""CIFAR-10 image-classification baseline.

``exp000`` trains a pre-activation ResNet with AdamW and cosine decay to
roughly 94% test accuracy. Later experiments layer on the "speedrun" recipe
(PCA whitening, Muon, test-time augmentation) one change at a time.
"""
