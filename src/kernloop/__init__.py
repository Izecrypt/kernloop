"""KernLoop: an autonomous GPU kernel engineer.

The agent loop: propose (Nemotron via Nebius Token Factory), validate,
benchmark on a backend (Nebius Serverless Jobs, or a local fake for testing),
promote only on statistically real improvement, and log everything.
"""

__version__ = "0.1.0"
