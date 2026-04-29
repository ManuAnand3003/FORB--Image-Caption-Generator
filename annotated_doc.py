"""Compatibility shim for FastAPI's optional annotated_doc dependency.

Some environments ship an empty or incomplete annotated_doc package. FastAPI
only needs the Doc metadata wrapper at import time, so this local module keeps
`python main.py` working without changing the installed environment.
"""


class Doc(str):
    def __new__(cls, documentation: str = ""):
        return super().__new__(cls, documentation)