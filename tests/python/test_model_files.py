"""The model file (.hbm) and the results file (.hbr).

A model saved as a model file and read back solves to the same bits as the original: the
regression problems (lids, sheets, pressure surfaces, walls, a channel, interior surfaces,
B-spline and flat patches, several bodies, field and body points) and bodies with generalized
modes, dipole dampers, analytic patches and a trimmed patch body. An existing file is kept unless
overwrite; a file that is not a model file, damaged or cut short, or of another version of the
model record is refused. Results saved to a results file read back with the same arrays and the
model solved, which a model file also reads. Meshes are procedural (tests/python/meshes.py)."""

import struct
import zlib

import numpy as np
import pytest

import hydrobem as hb

from regression_problems import PROBLEMS, same_outputs

G = 9.80665


def _round_trip(m, tmp_path):
    m.save(tmp_path / "case.hbm")
    return hb.Model.load(tmp_path / "case.hbm")


def test_round_trip_solves_the_same(tmp_path):
    # a smoke test of the solve after a round trip: one small case with field and body points (every
    # regression and feature case is saved and loaded to the same tree, bit for bit, by
    # test_model_document.py, and the solve reads nothing outside the tree: its
    # test_the_model_and_its_model_file_solve_the_same)
    m = PROBLEMS["lo_cylinder_field_body_points"]()
    same_outputs(m.solve(), _round_trip(m, tmp_path).solve())


def test_results_file(tmp_path):
    m = PROBLEMS["lo_cylinder_drift"]()
    r = m.solve()
    r.save(tmp_path / "run.hbr")
    r.save(tmp_path / "new" / "run.hbr")
    assert (tmp_path / "new" / "run.hbr").exists()  # the folder made by the engine
    s = hb.load_results(tmp_path / "run.hbr")  # its arrays bit for bit: test_results_file.py
    assert s.route == r.route
    q = s.model
    assert q.bodies.names() == m.bodies.names() and np.array_equal(q.environment.omegas, m.environment.omegas)
    same_outputs(r, q.solve())  # the case in the file solves to the same bits
    with pytest.raises(FileExistsError, match=r"run\.hbr exists"):
        r.save(tmp_path / "run.hbr")


def test_model_file(tmp_path):
    # one file with the meshes inside; the folder made; an existing file kept unless overwrite
    m = PROBLEMS["lo_cylinder_drift"]()
    m.save(tmp_path / "a" / "case.hbm")
    assert sorted(f.name for f in (tmp_path / "a").iterdir()) == ["case.hbm"]
    with pytest.raises(FileExistsError, match=r"case\.hbm exists"):
        m.save(tmp_path / "a" / "case.hbm")
    m.save(tmp_path / "a" / "case.hbm", overwrite=True)
    assert str(hb.Model.load(tmp_path / "a" / "case.hbm")) == str(m)
    # the inputs of a results file
    m.solve()
    m.save_results(tmp_path / "run.hbr")
    assert str(hb.Model.load(tmp_path / "run.hbr")) == str(m)
    # a model file holds no results
    with pytest.raises(ValueError, match="no results|model file"):
        hb.load_results(tmp_path / "a" / "case.hbm")


def test_damaged_model_files_are_refused(tmp_path):
    m = PROBLEMS["lo_cylinder_drift"]()
    f = tmp_path / "case.hbm"
    m.save(f)
    data = f.read_bytes()
    (tmp_path / "text.hbm").write_text('format = "hydrobem-case"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a HydroBEM model or results file"):
        hb.Model.load(tmp_path / "text.hbm")
    (tmp_path / "cut.hbm").write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="cut short or damaged"):
        hb.Model.load(tmp_path / "cut.hbm")
    flipped = bytearray(data)
    flipped[200] ^= 0xFF  # inside the first entry
    (tmp_path / "flip.hbm").write_bytes(bytes(flipped))
    with pytest.raises(ValueError, match="damaged"):
        hb.Model.load(tmp_path / "flip.hbm")


def _with_model_record_version(data: bytes, version: int) -> bytes:
    """A model file with the version of its model record changed, its checksums made anew (the
    file layout: a 64-byte header, the entries, the directory at the end)."""
    out = bytearray(data)
    dir_offset, dir_size = struct.unpack_from("<QQ", out, 16)
    d = dir_offset + 4
    for _ in range(struct.unpack_from("<I", out, dir_offset)[0]):
        n = struct.unpack_from("<I", out, d)[0]
        name = bytes(out[d + 4:d + 4 + n]).decode()
        d += 4 + n + 1
        rank = out[d]
        d += 1 + 8 * rank
        offset, size = struct.unpack_from("<QQ", out, d)
        d += 16
        if name == "model":
            struct.pack_into("<I", out, offset, version)
            struct.pack_into("<I", out, d, zlib.crc32(out[offset:offset + size]))
        d += 4
        for _ in range(2):  # the dims and the unit
            d += 4 + struct.unpack_from("<I", out, d)[0]
    struct.pack_into("<I", out, 32, zlib.crc32(out[dir_offset:dir_offset + dir_size]))
    return bytes(out)


RECORD_VERSION = 13  # the version of the model record this HydroBEM writes


def test_a_model_record_of_another_version_is_refused(tmp_path):
    # no conversion of an earlier layout of the model: the version is named
    m = PROBLEMS["lo_cylinder_drift"]()
    m.save(tmp_path / "case.hbm")
    data = (tmp_path / "case.hbm").read_bytes()
    (tmp_path / "same.hbm").write_bytes(_with_model_record_version(data, RECORD_VERSION))
    assert str(hb.Model.load(tmp_path / "same.hbm")) == str(m)  # the checksums made as the engine makes them
    (tmp_path / "v1.hbm").write_bytes(_with_model_record_version(data, 1))
    with pytest.raises(ValueError, match=rf"v1\.hbm: model: model record version 1; this HydroBEM reads version "
                                         rf"{RECORD_VERSION} \(no"):  # fmt: skip
        hb.Model.load(tmp_path / "v1.hbm")
    # a file of the version before (its mesh items trim_records and source_panels) is refused alike
    (tmp_path / "old.hbm").write_bytes(_with_model_record_version(data, RECORD_VERSION - 1))
    with pytest.raises(ValueError, match=rf"old\.hbm: model: model record version {RECORD_VERSION - 1}; this"):
        hb.Model.load(tmp_path / "old.hbm")
    m.solve()
    m.save_results(tmp_path / "run.hbr")
    (tmp_path / "v1.hbr").write_bytes(_with_model_record_version((tmp_path / "run.hbr").read_bytes(), 1))
    with pytest.raises(ValueError, match="model record version 1"):
        _ = hb.load_results(tmp_path / "v1.hbr").model


def test_tool_text_is_kept(tmp_path):
    # the data of a client of the engine (a graphical interface, a script): any text, kept
    m = PROBLEMS["lo_cylinder_drift"]()
    text = '{"window": [800, 600], "view": "top"}\nline two ✓'
    m.tool.add("gui", text=text)
    q = _round_trip(m, tmp_path)
    assert q.tool.names() == ["gui"] and q.tool["gui"].text == text


def test_a_model_that_cannot_be_read_back_is_not_kept(tmp_path):
    # changing a mesh to a surface mesh checks its panels as giving them would, so no model
    # is saved that its file would refuse
    import numpy as np

    m = hb.Model()
    flat = m.meshes.add("f", kind="flat_patches", vertices=np.zeros((1, 4, 3)))
    with pytest.raises(ValueError, match="fewer than three distinct vertices"):
        flat.kind = "panels"
    assert flat.kind == "flat_patches"
    m.save(tmp_path / "ok.hbm")
    assert hb.Model.load(tmp_path / "ok.hbm").meshes["f"].kind == "flat_patches"


def test_results_file_read_after_the_working_folder_changes(tmp_path, monkeypatch):
    # the quantities are read when first used, by the absolute path of the file
    from bodies import lo_free_cylinder
    from problems import model

    r = model(lo_free_cylinder(), [1.0], [0.0]).solve()
    monkeypatch.chdir(tmp_path)
    r.save("run.hbr")
    back = hb.load_results("run.hbr")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert back.added_mass.tobytes() == r.added_mass.tobytes()
