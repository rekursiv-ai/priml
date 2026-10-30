# Only the symbols this repo imports. harbor 0.24.0 is NOT a priml dependency:
# the agent that subclasses BaseAgent runs inside Harbor's own trial process,
# which provides the package. These stubs pin the 0.24.0 API surface that
# priml/baselines/openaimath/scripts/agent.py builds against, so both type
# checkers can verify that subclass without installing Harbor.
#
# A `.pyi` SHADOWS the module, so anything omitted here goes unknown for every
# consumer. A new import from this package must be added here too.
