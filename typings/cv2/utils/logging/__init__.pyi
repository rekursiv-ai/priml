__all__: list[str] = ...
LOG_LEVEL_SILENT: int
LOG_LEVEL_FATAL: int
LOG_LEVEL_ERROR: int
LOG_LEVEL_WARNING: int
LOG_LEVEL_INFO: int
LOG_LEVEL_DEBUG: int
LOG_LEVEL_VERBOSE: int
ENUM_LOG_LEVEL_FORCE_INT: int
LogLevel = int

def getLogLevel() -> LogLevel: ...
def setLogLevel(logLevel: LogLevel) -> LogLevel: ...
