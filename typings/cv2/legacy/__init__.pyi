import typing as _typing

import cv2
import cv2.typing

__all__: list[str] = ...

class Tracker(cv2.Algorithm):
    @_typing.overload
    def init(
        self,
        image: cv2.typing.MatLike,
        boundingBox: cv2.typing.Rect2d,
    ) -> bool: ...
    @_typing.overload
    def init(self, image: cv2.UMat, boundingBox: cv2.typing.Rect2d) -> bool: ...
    @_typing.overload
    def update(self, image: cv2.typing.MatLike) -> tuple[bool, cv2.typing.Rect2d]: ...
    @_typing.overload
    def update(self, image: cv2.UMat) -> tuple[bool, cv2.typing.Rect2d]: ...

class TrackerMIL(Tracker):
    @classmethod
    def create(cls) -> TrackerMIL: ...

class TrackerBoosting(Tracker):
    @classmethod
    def create(cls) -> TrackerBoosting: ...

class TrackerMedianFlow(Tracker):
    @classmethod
    def create(cls) -> TrackerMedianFlow: ...

class TrackerTLD(Tracker):
    @classmethod
    def create(cls) -> TrackerTLD: ...

class TrackerKCF(Tracker):
    @classmethod
    def create(cls) -> TrackerKCF: ...

class TrackerMOSSE(Tracker):
    @classmethod
    def create(cls) -> TrackerMOSSE: ...

class MultiTracker(cv2.Algorithm):
    def __init__(self) -> None: ...
    @_typing.overload
    def add(
        self,
        newTracker: Tracker,
        image: cv2.typing.MatLike,
        boundingBox: cv2.typing.Rect2d,
    ) -> bool: ...
    @_typing.overload
    def add(
        self,
        newTracker: Tracker,
        image: cv2.UMat,
        boundingBox: cv2.typing.Rect2d,
    ) -> bool: ...
    @_typing.overload
    def update(
        self,
        image: cv2.typing.MatLike,
    ) -> tuple[bool, _typing.Sequence[cv2.typing.Rect2d]]: ...
    @_typing.overload
    def update(
        self,
        image: cv2.UMat,
    ) -> tuple[bool, _typing.Sequence[cv2.typing.Rect2d]]: ...
    def getObjects(self) -> _typing.Sequence[cv2.typing.Rect2d]: ...
    @classmethod
    def create(cls) -> MultiTracker: ...

class TrackerCSRT(Tracker):
    @classmethod
    def create(cls) -> TrackerCSRT: ...
    @_typing.overload
    def setInitialMask(self, mask: cv2.typing.MatLike) -> None: ...
    @_typing.overload
    def setInitialMask(self, mask: cv2.UMat) -> None: ...

def upgradeTrackingAPI(legacy_tracker: Tracker) -> Tracker: ...
