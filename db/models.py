"""
SQLAlchemy ORM models for persistence layer.
Only Automation is persisted - triggers, conditions, and actions are registry references
stored as JSON within the automation record.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, String

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class AutomationModel(Base):
    """
    Database model for Automation.
    Only model persisted in the database. Trigger, conditions, and actions are stored
    as JSON fields since they reference registry types, not separate entities.
    """

    __tablename__ = "automations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    
    # Trigger stored as JSON: {"type": "...", "config": {...}}
    trigger = Column(JSON, nullable=False)
    
    # Conditions stored as JSON: [[{"type": "...", "params": {...}}, ...], ...]
    # Outer list = OR groups, inner list = AND conditions
    conditions = Column(JSON, default=list, nullable=False)
    
    # Actions stored as JSON: [{"type": "...", "params": {...}}, ...]
    actions = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<AutomationModel(id={self.id}, name={self.name})>"
