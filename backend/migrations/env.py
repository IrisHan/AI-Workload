from alembic import context
from app.infrastructure.database import Base
from sqlalchemy import create_engine

config = context.config


def run(connection):
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    connection = config.attributes.get("connection")
    if connection is not None:
        run(connection)
    else:
        import os

        engine = create_engine(os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url")))
        with engine.connect() as connection:
            run(connection)
