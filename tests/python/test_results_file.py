"""The results file of the engine, .hbr: every quantity, the model, the meta data and
the diagnostics written and read back, and the checks of a damaged file, through the engine's
results (``_Results``: the results of a model of the extension module, as the Python package
holds them)."""

import struct
import zlib

import pytest

import hydrobem as hb
from hydrobem import _core
from hydrobem.queries import quantity_names

from problems import FEATURES
from regression_problems import PROBLEMS, same_outputs


def _loaded(m, folder):
    m.save(folder / "case.hbm")
    return _core.Model.load(folder / "case.hbm")


class _Results:
    """Holds the engine's results (hydrobem._core.Results)."""

    def __init__(self, r):
        self.r = r

    @classmethod
    def solve(cls, doc):
        """A copy of the results of a solve of the model."""
        return cls(doc.solve())

    @classmethod
    def open(cls, path):
        """The results of the calculated model of a results file."""
        return cls(_core.Model.load_results(path).results())

    def save(self, path, overwrite=False):
        self.r.save(path, overwrite)

    def quantities(self):
        return self.r.route, {quantity_names()[k]: self.r.quantity(k) for k in self.r.available()}

    def meta(self):
        return self.r.meta()

    def model(self):
        return self.r.model()


def _model_text(doc):
    """The model as print shows it: every object and item given."""
    return str(hb.Model(doc))


_NAMES = ("lo_cylinder_drift", "lo_cylinder_field_body_points", "lo_moonpool_lid", "lo_acv_two_modes",
          "lo_two_bodies", "ho_spline_cylinder_drift")  # fmt: skip
CASES = {name: PROBLEMS[name] for name in _NAMES}
CASES.update({f"feature_{name}": build for name, build in FEATURES.items()})


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """Each case solved once for the module: its engine model, its results and their saved .hbr
    (the tests read them and never change them)."""
    cache = {}

    def get(name):
        if name not in cache:
            folder = tmp_path_factory.mktemp(name)
            doc = _loaded(CASES[name](), folder)
            r = _Results.solve(doc)
            r.save(folder / "run.hbr")
            cache[name] = doc, r, folder / "run.hbr"
        return cache[name]

    return get


@pytest.mark.parametrize("name", list(CASES))
def test_a_saved_results_file_opens_to_the_same_results(name, solved, tmp_path):
    # an IO rule: every quantity of each kind of problem written and read back bit for bit
    doc, r, path = solved(name)
    o = _Results.open(path)
    route, q = r.quantities()
    route2, q2 = o.quantities()
    assert route2 == route
    same_outputs(q, q2)
    assert o.r.diagnostics() == r.r.diagnostics()
    # the model of the file is the model solved
    assert _model_text(o.model()) == _model_text(doc)
    meta = o.meta()
    assert meta["route"] == route and meta["time"].endswith("Z") and len(meta["engine_seconds"]) >= 1
    assert meta["environment_hooks"] == ()  # no hook of the environment set in the tests
    # saved again from the opened handle (its quantities read from the first file)
    o.save(tmp_path / "again.hbr")
    same_outputs(q, _Results.open(tmp_path / "again.hbr").quantities()[1])


def _layout(path):
    b = path.read_bytes()
    version, flags, dir_offset, dir_size, dir_crc = struct.unpack_from("<IIQQI", b, 8)
    return b, version, dir_offset, dir_size, dir_crc


def _entries(b, dir_offset):
    at = dir_offset
    (n,) = struct.unpack_from("<I", b, at)
    at += 4
    out = []
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", b, at)
        name = b[at + 4:at + 4 + ln].decode()
        at += 4 + ln
        typ, rank = b[at], b[at + 1]
        at += 2 + 8 * rank
        offset, size, crc = struct.unpack_from("<QQI", b, at)
        at += 20
        for _ in range(2):
            (ln,) = struct.unpack_from("<I", b, at)
            at += 4 + ln
        out.append((name, typ, offset, size, crc))
    return out


def test_the_layout_and_its_crc(solved):
    b, version, dir_offset, dir_size, dir_crc = _layout(solved("lo_cylinder_drift")[2])
    assert b[:8] == b"\x89HBR\r\n\x1a\n" and version == 5
    assert dir_offset + dir_size == len(b)
    assert zlib.crc32(b[dir_offset:]) == dir_crc  # the CRC-32 of zlib
    entries = _entries(b, dir_offset)
    names = [e[0] for e in entries]
    assert names[:3] == ["meta", "diagnostics", "model"] and "q/added_mass" in names
    for name, _, offset, size, crc in entries:
        assert offset % 64 == 0 and zlib.crc32(b[offset:offset + size]) == crc, name


def test_a_damaged_file_is_refused(solved, tmp_path):
    _, r, path = solved("lo_cylinder_drift")
    b, _, dir_offset, _, _ = _layout(path)
    bad = tmp_path / "bad.hbr"
    for cut in (0, 10, 63, 64, dir_offset, len(b) - 1):
        bad.write_bytes(b[:cut])
        with pytest.raises(ValueError, match="not a HydroBEM model or results file|cut short|damaged"):
            _Results.open(bad)
    entry = next(e for e in _entries(b, dir_offset) if e[0] == "q/added_mass")
    c = bytearray(b)
    c[entry[2] + 5] ^= 0x10
    bad.write_bytes(bytes(c))
    o = _Results.open(bad)  # opened: the quantities are read when asked for
    with pytest.raises(ValueError, match="q/added_mass is damaged"):
        o.quantities()
    c = bytearray(b)
    c[dir_offset + 8] ^= 0x01
    bad.write_bytes(bytes(c))
    with pytest.raises(ValueError, match="directory is damaged"):
        _Results.open(bad)
    # an entry whose element type or size is not the quantity's, with a valid directory CRC
    for at, value, message in ((0, 4, "another element type"), (1 + 1 + 3 * 8 + 8, 1, "does not match its shape")):
        d = bytearray(b[dir_offset:])
        name = b"q/added_mass"
        k = d.index(struct.pack("<I", len(name)) + name) + 4 + len(name)
        if at == 0:
            d[k] = value  # T_FLOAT64 instead of T_FLOAT32
        else:
            (size,) = struct.unpack_from("<Q", d, k + at)
            struct.pack_into("<Q", d, k + at, size - 4)
        head = bytearray(b[:64])
        struct.pack_into("<I", head, 32, zlib.crc32(bytes(d)))
        bad.write_bytes(bytes(head) + b[64:dir_offset] + bytes(d))
        with pytest.raises(ValueError, match=message):
            _Results.open(bad)
    c = bytearray(b)
    for version in (4, 6):  # the format before the result tree, and a newer one
        c[8] = version
        bad.write_bytes(bytes(c))
        with pytest.raises(ValueError, match=f"file format version {version}; this HydroBEM reads 5"):
            _Results.open(bad)
    with pytest.raises(OSError, match="exists"):
        r.save(path)


def test_an_unknown_entry_is_ignored(solved, tmp_path):
    _, r, path = solved("lo_cylinder_drift")
    b, version, dir_offset, dir_size, _ = _layout(path)
    # a later writer's entry: appended to the data, listed first in the directory
    data = b[:dir_offset]
    extra = b"a later entry"
    new_offset = len(data)
    data += extra + b"\0" * (-(len(data) + len(extra)) % 64)
    d = b[dir_offset:]
    (n,) = struct.unpack_from("<I", d, 0)
    name = b"x/later"
    e = struct.pack("<I", len(name)) + name + bytes([0, 0]) + struct.pack("<QQI", new_offset, len(extra),
                                                                            zlib.crc32(extra))  # fmt: skip
    e += struct.pack("<I", 0) + struct.pack("<I", 0)
    d = struct.pack("<I", n + 1) + e + d[4:]
    head = bytearray(b[:64])
    struct.pack_into("<QQI", head, 16, len(data), len(d), zlib.crc32(d))
    (tmp_path / "later.hbr").write_bytes(bytes(head) + data[64:] + d)
    same_outputs(r.quantities()[1], _Results.open(tmp_path / "later.hbr").quantities()[1])


def test_the_results_belong_to_the_inputs(solved, tmp_path):
    # a change discards the results; a clone stays; a results file loads as a calculated model
    doc = _loaded(PROBLEMS["lo_cylinder_drift"](), tmp_path / "in")
    assert doc.state == "reset"
    kept = _Results.solve(doc)
    assert doc.state == "calculated"
    doc.object("environment").set("density", 1025.0)
    assert doc.state == "reset"
    with pytest.raises(RuntimeError, match="no results"):
        doc.results()
    assert kept.quantities()[1]["added_mass"].size  # the clone is independent of the change
    _, first, path = solved("lo_cylinder_drift")
    n = hb.Model.load_results(path)
    added_mass = first.quantities()[1]["added_mass"]
    assert n.state == "calculated" and n.results.added_mass.tobytes() == added_mass.tobytes()
    n.environment.depth = 40.0
    assert n.state == "reset"
    with pytest.raises(RuntimeError, match="no results"):
        _ = n.results
