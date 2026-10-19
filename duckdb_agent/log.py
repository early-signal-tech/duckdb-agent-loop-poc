"""Logging for the agent: one named logger, plus console setup for scripts."""

import logging


# All agent output goes through this logger. Levels:
#   INFO    -- progress and metadata: iterations, stop reasons, tokens, latency, cost,
#              tool names and outcomes. No query data, SQL text, or answers.
#   DEBUG   -- the content too: SQL, thinking summaries, schema lookups, the final answer.
#   WARNING -- runs that end abnormally (API errors, crashes, truncated or refused answers).
# As a library it adds no handlers (only a NullHandler): an app or service configures
# logging its own way, and records propagate to the root logger. Scripts can call
# configure_logging() for the console output this agent used to print.
LOGGER_NAME = "duckdb_agent"


logger = logging.getLogger(LOGGER_NAME)


logger.addHandler(logging.NullHandler())


def configure_logging(verbose: bool = False, stream=None) -> logging.Logger:
    """Send agent logs to the console (stderr by default) as plain messages.

    verbose=True adds DEBUG content (SQL, thinking, the answer) -- the full trace a
    developer watches. Safe to call more than once: it replaces its own handler.
    Records stop at this logger, so a root handler doesn't print them twice.
    """
    for handler in [h for h in logger.handlers if getattr(h, "_agent_console", False)]:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._agent_console = True
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger
