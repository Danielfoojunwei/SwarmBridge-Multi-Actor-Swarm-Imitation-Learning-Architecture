"""Database models"""

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class CSARecord(Base):
    """CSA artifact record"""

    __tablename__ = "csa_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    skill_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(Text)

    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer)

    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    uploaded_by = Column(String(255))

    signature_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Metadata
    privacy_mode = Column(String(50))
    num_demonstrations = Column(Integer)
    training_sites = Column(Text)  # Comma-separated
    compatible_robots = Column(Text)  # Comma-separated
    test_pass_rate = Column(Float)

    # Relationships
    deployments = relationship("DeploymentRecord", back_populates="csa")


class DeploymentRecord(Base):
    """Deployment tracking record"""

    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, index=True)
    csa_id = Column(Integer, ForeignKey("csa_artifacts.id"), nullable=False)
    site_id = Column(String(255), nullable=False, index=True)

    deployed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    deployed_by = Column(String(255))

    status = Column(String(50), default="deployed")  # deployed, rolled_back, failed
    rolled_back_at = Column(DateTime)

    # Relationships
    csa = relationship("CSARecord", back_populates="deployments")
