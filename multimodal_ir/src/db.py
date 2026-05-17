# src/db.py
"""
SQLite database setup using SQLAlchemy.
Tables: papers, figures, embeddings_meta
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Float, ForeignKey, Boolean, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pathlib import Path
from src.config import cfg

Base = declarative_base()


class Paper(Base):
    __tablename__ = "papers"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    paper_id    = Column(String(64), unique=True, nullable=False, index=True)
    title       = Column(Text, nullable=False)
    abstract    = Column(Text)
    authors     = Column(JSON)          # list of author name strings
    year        = Column(Integer)
    venue       = Column(String(256))
    pdf_path    = Column(String(512))
    doi         = Column(String(256))
    # Embedding index position (row in the numpy array / FAISS index)
    text_faiss_id = Column(Integer, unique=True)

    figures = relationship("Figure", back_populates="paper", cascade="all, delete-orphan")
    pages = relationship("DocumentPage", back_populates="paper", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
        }


class Figure(Base):
    __tablename__ = "figures"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    paper_id        = Column(String(64), ForeignKey("papers.paper_id"), nullable=False, index=True)
    figure_number   = Column(Integer)
    caption         = Column(Text)
    image_path      = Column(String(512))   # relative path under data/figures/
    page_number     = Column(Integer)
    width           = Column(Integer)
    height          = Column(Integer)
    # Embedding index position in FAISS figure index
    figure_faiss_id = Column(Integer, unique=True)

    paper = relationship("Paper", back_populates="figures")

    def to_dict(self):
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "figure_number": self.figure_number,
            "caption": self.caption,
            "image_path": self.image_path,
        }


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    paper_id      = Column(String(64), ForeignKey("papers.paper_id"), nullable=False, index=True)
    page_number   = Column(Integer, nullable=False)
    text          = Column(Text)
    image_path    = Column(String(512))
    text_faiss_id = Column(Integer, unique=True)

    paper = relationship("Paper", back_populates="pages")

    def to_dict(self):
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "page_number": self.page_number,
            "text": self.text,
            "image_path": self.image_path,
        }


def get_engine(db_path: str = None):
    path = db_path or cfg.data.db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=False)


def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(db_path: str = None):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
