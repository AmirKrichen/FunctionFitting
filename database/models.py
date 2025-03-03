from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import Float, String, MetaData, create_engine


class Base(DeclarativeBase):
    """
    Serves as the base class for the database models.
    """
    pass


class TrainData(Base):
    """
    Metadata for the train_data table in the database.
    """
    __tablename__ = "train_data"
    x: Mapped[float] = mapped_column(Float, primary_key=True)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    y3: Mapped[float] = mapped_column(Float, nullable=False)
    y4: Mapped[float] = mapped_column(Float, nullable=False)


class IdealFunctions(Base):
    """
    Metadata for the ideal_data table in the database.
    """
    __tablename__ = "ideal_functions"
    x: Mapped[float] = mapped_column(Float, primary_key=True)
    for i in range(1, 51):
        locals()[f"y{i}"]: Mapped[float] = mapped_column(Float, nullable=False)
    # We use the locals() function here to name our columns dynamically
    # Columns range from y1 to y50


class TestData(Base):
    """
    Metadata for the test_data table in the database.
    """
    __tablename__ = "test_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    delta_y: Mapped[float] = mapped_column(Float, nullable=True)
    ideal_function: Mapped[str] = mapped_column(String, nullable=True)
    # we need to add an id column to test test table as it has repeated values
    # SQLAlchemy's Declarative Mapping requires a primary key column


def create_session(database_reset=False):
    """
    Creates and returns a session for the specified database.

    :return: Session: An instance of SQLAlchemy's `Session` object, used for
            interacting with the database.
    """
    engine = create_engine("sqlite:///database.db")

    if database_reset:
        # Finds and drops all tables from the specified database
        metadata = MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(engine)
        print("Database initiation was successful.")
    else:
        Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    return Session()
