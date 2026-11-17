"""Versioned, full logical migration of Planner data; never copy a live DB file."""

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
TABLES = (
    "model_versions", "hardware_versions", "model_structure_snapshots",
    "workload_designs", "workload_design_revisions", "workload_runs",
    "workload_plans", "workload_plan_details", "alembic_version",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def quote(name):
    return '"' + name.replace('"', '""') + '"'


@contextmanager
def readonly(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.rollback()
        connection.close()


def tables(connection):
    return {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def structure(connection, name):
    columns = connection.execute(f"PRAGMA table_info({quote(name)})").fetchall()
    indexes = []
    for row in connection.execute(f"PRAGMA index_list({quote(name)})"):
        index_columns = [item[2] for item in connection.execute(f"PRAGMA index_info({quote(row[1])})")]
        indexes.append({"unique": row[2], "origin": row[3], "partial": row[4], "columns": index_columns})
    return {
        "columns": [list(row[1:]) for row in columns],
        "indexes": sorted(indexes, key=canonical),
        "foreign_keys": sorted(
            [list(row[2:]) for row in connection.execute(f"PRAGMA foreign_key_list({quote(name)})")],
            key=canonical,
        ),
    }


def semantic(row, json_columns):
    # SQL NULL stays None; JSON null remains a separate tagged value.
    return {key: {"json_value": json.loads(value)} if key in json_columns and value is not None
            else value for key, value in row.items()}


def sqlite_stats(connection, metadata):
    result = {}
    for name in TABLES:
        meta = metadata[name]
        columns = meta["columns"]
        order = ",".join(map(quote, meta["primary_key"]))
        checksum = hashlib.sha256()
        count = 0
        for values in connection.execute(f"SELECT * FROM {quote(name)} ORDER BY {order}"):
            row = dict(zip(columns, values, strict=True))
            checksum.update((canonical(semantic(row, meta["json_columns"])) + "\n").encode())
            count += 1
        result[name] = {"rows": count, "content_sha256": checksum.hexdigest()}
    return result


def check_sqlite(connection):
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("SQLite integrity_check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("SQLite foreign_key_check failed")
    if tables(connection) != set(TABLES):
        raise ValueError(f"Unexpected database tables: {sorted(tables(connection))}")


def export_sqlite(source, output):
    if output.exists():
        raise ValueError("Export output must not exist")
    output.mkdir(parents=True, mode=0o700)
    with readonly(source) as connection:
        check_sqlite(connection)
        version = connection.execute("SELECT version_num FROM alembic_version").fetchall()
        if version != [("0003",)]:
            raise ValueError(f"Expected migration 0003, found {version}")
        active = connection.execute(
            "SELECT count(*) FROM workload_runs WHERE status IN ('queued','running','cancel_requested')"
        ).fetchone()[0]
        if active:
            raise ValueError("Active Workload runs present; wait for terminal state before export")
        metadata = {}
        for name in TABLES:
            info = connection.execute(f"PRAGMA table_info({quote(name)})").fetchall()
            metadata[name] = {
                "columns": [item[1] for item in info],
                "json_columns": [item[1] for item in info if item[2].upper() == "JSON"],
                "primary_key": [item[1] for item in sorted(info, key=lambda item: item[5]) if item[5]],
                "structure": structure(connection, name),
            }
        for name in TABLES:
            meta = metadata[name]
            checksum = hashlib.sha256()
            content = hashlib.sha256()
            count = 0
            filename = f"{name}.ndjson.gz"
            with gzip.open(output / filename, "wb") as stream:
                order = ",".join(map(quote, meta["primary_key"]))
                for values in connection.execute(f"SELECT * FROM {quote(name)} ORDER BY {order}"):
                    row = dict(zip(meta["columns"], values, strict=True))
                    line = (canonical(row) + "\n").encode()
                    stream.write(line)
                    checksum.update(line)
                    content.update((canonical(semantic(row, meta["json_columns"])) + "\n").encode())
                    count += 1
            meta.update(file=filename, rows=count, ndjson_sha256=checksum.hexdigest(),
                        content_sha256=content.hexdigest(), file_sha256=digest(output / filename))
        ddl = "\n".join(row[0] + ";" for row in connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' AND type IN ('table','index') "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"))
        (output / "schema.sql").write_text(ddl + "\n")
        with gzip.open(output / "full.sql.gz", "wt", encoding="utf-8") as stream:
            for line in connection.iterdump():
                stream.write(line + "\n")
        manifest = {
            "format": "nsight-logical-1", "created_at": datetime.now().astimezone().isoformat(),
            "source": "SQLite read-only transaction including committed WAL",
            "schema_version": "0003", "table_order": list(TABLES), "tables": metadata,
            "files": {name: digest(output / name) for name in ("schema.sql", "full.sql.gz")},
            "integrity_check": "ok", "foreign_key_check": [], "active_runs": active,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in output.iterdir():
        path.chmod(0o600)
    return manifest


def load_bundle(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest["format"] != "nsight-logical-1" or manifest["schema_version"] != "0003"
            or manifest["table_order"] != list(TABLES) or set(manifest["tables"]) != set(TABLES)):
        raise ValueError("Unsupported or incomplete migration bundle")
    for name, expected in manifest["files"].items():
        if name not in ("schema.sql", "full.sql.gz") or digest(bundle / name) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    for name, meta in manifest["tables"].items():
        if meta["file"] != f"{name}.ndjson.gz" or digest(bundle / meta["file"]) != meta["file_sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
        checksum = hashlib.sha256()
        count = 0
        with gzip.open(bundle / meta["file"], "rb") as stream:
            for line in stream:
                checksum.update(line)
                row = json.loads(line)
                if set(row) != set(meta["columns"]):
                    raise ValueError(f"Invalid row columns: {name}")
                count += 1
        if checksum.hexdigest() != meta["ndjson_sha256"] or count != meta["rows"]:
            raise ValueError(f"Bundle row/hash mismatch: {name}")
    return manifest


def initialize(url, seed_data=False):
    from app.infrastructure.database import database, migrate

    engine, sessions = database(url)
    try:
        migrate(engine)
        if seed_data:
            from app.application.planner_service import seed
            from app.hardware.repository import HardwareRepository
            from app.repositories.catalog import CatalogRepository
            with sessions() as session:
                seed(CatalogRepository(session))
                HardwareRepository(session).seed()
    finally:
        engine.dispose()


def assert_stats(actual, metadata):
    expected = {name: {"rows": meta["rows"], "content_sha256": meta["content_sha256"]}
                for name, meta in metadata.items()}
    if actual != expected:
        raise ValueError("Restored table counts or content hashes differ from source")


def verify_sqlite(path, manifest):
    with readonly(path) as connection:
        check_sqlite(connection)
        for name, meta in manifest["tables"].items():
            if structure(connection, name) != meta["structure"]:
                raise ValueError(f"Restored schema differs from source: {name}")
        result = sqlite_stats(connection, manifest["tables"])
        assert_stats(result, manifest["tables"])
        return result


def restore_sqlite(bundle, target, manifest):
    if target.exists():
        raise ValueError("SQLite restore refuses an existing target file")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nsight-restore-", dir=target.parent) as folder:
        temporary = Path(folder) / "restored.db"
        initialize(f"sqlite:///{temporary.resolve()}")
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            for name in TABLES:
                if name != "alembic_version" and connection.execute(
                        f"SELECT count(*) FROM {quote(name)}").fetchone()[0]:
                    raise ValueError(f"Target table is not empty: {name}")
            connection.execute("DELETE FROM alembic_version")
            for name in TABLES:
                meta = manifest["tables"][name]
                placeholders = ",".join("?" for _ in meta["columns"])
                statement = f"INSERT INTO {quote(name)} VALUES ({placeholders})"
                with gzip.open(bundle / meta["file"], "rt", encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        connection.execute(statement, [row[column] for column in meta["columns"]])
            check_sqlite(connection)
            assert_stats(sqlite_stats(connection, manifest["tables"]), manifest["tables"])
        result = verify_sqlite(temporary, manifest)
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode=DELETE")
        temporary.chmod(0o600)
        # Atomic no-overwrite publication, even if another process creates target.
        os.link(temporary, target)
        return result


def pg_stats(connection, manifest, metadata):
    import sqlalchemy as sa

    result = {}
    for name in TABLES:
        meta = manifest["tables"][name]
        table = metadata.tables[name]
        expressions = [
            sa.cast(table.c[column], sa.Text).label(column) if column in meta["json_columns"]
            else table.c[column] for column in meta["columns"]
        ]
        query = sa.select(*expressions).order_by(*(table.c[key] for key in meta["primary_key"]))
        checksum = hashlib.sha256()
        count = 0
        for row in connection.execute(query).mappings():
            checksum.update((canonical(semantic(dict(row), meta["json_columns"])) + "\n").encode())
            count += 1
        result[name] = {"rows": count, "content_sha256": checksum.hexdigest()}
    assert_stats(result, manifest["tables"])
    return result


def postgres(bundle, url, manifest, restore=False):
    import sqlalchemy as sa

    engine = sa.create_engine(url)
    try:
        if engine.dialect.name != "postgresql":
            raise ValueError("DATABASE_URL must use PostgreSQL, or supply --sqlite explicitly")
        metadata = sa.MetaData()
        with engine.begin() as connection:
            metadata.reflect(bind=connection)
            if set(metadata.tables) != set(TABLES):
                raise ValueError("PostgreSQL schema table set differs; run init on a fresh database")
            if restore:
                connection.execute(sa.text("LOCK TABLE " + ",".join(map(quote, TABLES)) + " IN EXCLUSIVE MODE"))
                for name in TABLES:
                    if name != "alembic_version" and connection.scalar(
                            sa.select(sa.func.count()).select_from(metadata.tables[name])):
                        raise ValueError(f"Target table is not empty: {name}")
                connection.execute(metadata.tables["alembic_version"].delete())
                for name in TABLES:
                    meta = manifest["tables"][name]
                    table = metadata.tables[name]
                    with gzip.open(bundle / meta["file"], "rt", encoding="utf-8") as stream:
                        for line in stream:
                            row = json.loads(line)
                            for column in meta["json_columns"]:
                                raw = row[column]
                                row[column] = sa.null() if raw is None else (
                                    sa.JSON.NULL if raw == "null" else json.loads(raw))
                            # values(), rather than executemany, supports explicit SQL NULL.
                            connection.execute(table.insert().values(**row))
            return pg_stats(connection, manifest, metadata)
    finally:
        engine.dispose()


def create_postgres(name, owner=None):
    import sqlalchemy as sa
    from psycopg import sql

    url = os.environ.get("ADMIN_DATABASE_URL")
    if not url:
        raise ValueError("ADMIN_DATABASE_URL is required")
    engine = sa.create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        if engine.dialect.name != "postgresql":
            raise ValueError("ADMIN_DATABASE_URL must use PostgreSQL")
        with engine.connect() as connection:
            with connection.connection.driver_connection.cursor() as cursor:
                statement = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
                if owner:
                    statement += sql.SQL(" OWNER {}").format(sql.Identifier(owner))
                cursor.execute(statement)
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export")
    export.add_argument("--sqlite", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    create = commands.add_parser("create-postgres")
    create.add_argument("--name", required=True)
    create.add_argument("--owner", help="Existing PostgreSQL role that owns the new database")
    for name in ("init", "restore", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--sqlite", type=Path)
        if name == "init":
            command.add_argument("--seed", action="store_true")
        else:
            command.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "create-postgres":
        create_postgres(args.name, args.owner)
        print("PostgreSQL database created; run init without --seed before full restore")
    elif args.action == "export":
        manifest = export_sqlite(args.sqlite, args.output)
        print(json.dumps({name: meta["rows"] for name, meta in manifest["tables"].items()}, indent=2))
    elif args.action == "init":
        if args.sqlite:
            args.sqlite.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{args.sqlite.resolve()}" if args.sqlite else os.environ["DATABASE_URL"]
        initialize(url, args.seed)
        print("Alembic head initialized" + (" with seeds" if args.seed else " without seeds"))
    else:
        manifest = load_bundle(args.bundle)
        if args.sqlite:
            result = restore_sqlite(args.bundle, args.sqlite, manifest) if args.action == "restore" else (
                verify_sqlite(args.sqlite, manifest))
        else:
            result = postgres(args.bundle, os.environ["DATABASE_URL"], manifest, args.action == "restore")
        print(json.dumps({"action": args.action, "verified": result}, indent=2))


if __name__ == "__main__":
    main()
