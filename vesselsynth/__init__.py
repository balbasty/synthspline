import interpol.backend
try:
    import distmap.backend
except ImportError:
    distmap = None


class _Backend:

    def __init__(self) -> None:
        self._jitfields = interpol.backend.jitfields
        if distmap:
            distmap.backend.jitfields = self._jitfields

    @property
    def jitfields(self):
        return self._jitfields

    @jitfields.setter
    def jitfields(self, value):
        self._jitfields = value
        interpol.backend.jitfields = value
        if distmap:
            distmap.backend.jitfields = value


backend = _Backend()
