import interpol


class _Backend:

    def __init__(self) -> None:
        self._jitfields = interpol.backend.jitfields

    @property
    def jitfields(self):
        return self._jitfields

    @jitfields.setter
    def jitfields(self, value):
        self._jitfields = value
        interpol.backend.jitfields = value


backend = _Backend()