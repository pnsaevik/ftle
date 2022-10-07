class Fields:
    def __init__(self, funcdict):
        self._funcdict = dict()
        for k in funcdict:
            self[k] = funcdict[k]

    def __getitem__(self, item):
        return self._funcdict[item]

    def __setitem__(self, key, value):
        self._funcdict[key] = value

    def keys(self):
        return self._funcdict.keys()

    def values(self):
        return self._funcdict.values()

    def items(self):
        return self._funcdict.items()

    @staticmethod
    def from_dict(funcdict):
        return Fields(funcdict)
