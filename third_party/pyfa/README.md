# Pyfa test database

This directory intentionally stores only the compressed Pyfa `eve.db` used by
tests and CI.

- Pyfa source is fetched in CI from `pyfa-org/Pyfa` at tag `v2.66.4`.
- `eve_db.zip` contains `eve.db` copied from a working Pyfa install.
- Uncompressed `eve.db` SHA256:
  `add72e0334e6ffe2d138c3ae2c2de1704d930a47aaa0e0ff082428a1256045df`
- Uncompressed `eve.db` size: `99368960` bytes.

The uncompressed database is not kept in the repository root because runtime
code expects it under `Pyfa-master/eve.db`, and `Pyfa-master/` remains ignored.
