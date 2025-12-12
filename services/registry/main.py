"""
CSA Registry Service

FastAPI service for managing Cooperative Skill Artifacts:
- Version management
- Signing and verification
- Deployment tracking
- Rollback support
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import engine, SessionLocal, Base
from .models import CSARecord, DeploymentRecord
from ...ml.artifact import CSALoader, CSAVerifier

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Dynamical-SIL CSA Registry",
    description="Cooperative Skill Artifact Registry with versioning and rollback",
    version="0.1.0",
)

logger = logging.getLogger(__name__)

# Storage directory
ARTIFACT_STORAGE = Path("/app/artifacts")
ARTIFACT_STORAGE.mkdir(parents=True, exist_ok=True)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic models
class CSAMetadataResponse(BaseModel):
    id: int
    skill_name: str
    version: str
    uploaded_at: datetime
    uploaded_by: str
    file_size: int
    signature_verified: bool
    is_active: bool

    class Config:
        from_attributes = True


class DeploymentRequest(BaseModel):
    csa_id: int
    site_id: str
    deployed_by: str


class RollbackRequest(BaseModel):
    site_id: str
    target_csa_id: int


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "csa-registry"}


# Upload CSA
@app.post("/api/v1/csa/upload", response_model=CSAMetadataResponse)
async def upload_csa(
    file: UploadFile = File(...),
    signature: UploadFile = File(None),
    uploaded_by: str = "unknown",
    db: Session = Depends(get_db),
):
    """
    Upload a new CSA artifact

    Args:
        file: CSA tarball
        signature: Detached signature file (optional)
        uploaded_by: Uploader identifier

    Returns:
        CSA metadata
    """
    # Save artifact
    artifact_path = ARTIFACT_STORAGE / file.filename
    with open(artifact_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Save signature if provided
    signature_verified = False
    if signature:
        sig_path = ARTIFACT_STORAGE / signature.filename
        with open(sig_path, "wb") as f:
            sig_contents = await signature.read()
            f.write(sig_contents)

        # Verify signature
        verifier = CSAVerifier()
        is_valid, message = verifier.verify_artifact(artifact_path, sig_path)
        signature_verified = is_valid
        logger.info(f"Signature verification: {message}")

    # Load and extract metadata
    loader = CSALoader()
    try:
        csa = loader.load(artifact_path, verify_checksums=True)
    except Exception as e:
        artifact_path.unlink()  # Clean up
        raise HTTPException(status_code=400, detail=f"Invalid CSA: {str(e)}")

    # Create database record
    csa_record = CSARecord(
        skill_name=csa.metadata.skill_name,
        version=csa.metadata.version,
        description=csa.metadata.description,
        file_path=str(artifact_path),
        file_size=artifact_path.stat().st_size,
        uploaded_by=uploaded_by,
        signature_verified=signature_verified,
        privacy_mode=csa.metadata.privacy_mode,
        num_demonstrations=csa.metadata.num_demonstrations,
        training_sites=",".join(csa.metadata.training_sites),
        compatible_robots=",".join(csa.metadata.compatible_robots),
        test_pass_rate=csa.metadata.test_pass_rate,
    )

    db.add(csa_record)
    db.commit()
    db.refresh(csa_record)

    logger.info(f"Uploaded CSA: {csa.metadata.skill_name} v{csa.metadata.version}")

    return CSAMetadataResponse.from_orm(csa_record)


# List CSAs
@app.get("/api/v1/csa/list", response_model=List[CSAMetadataResponse])
async def list_csas(
    skill_name: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db),
):
    """List CSA artifacts"""
    query = db.query(CSARecord)

    if skill_name:
        query = query.filter(CSARecord.skill_name == skill_name)

    if active_only:
        query = query.filter(CSARecord.is_active == True)

    csas = query.order_by(CSARecord.uploaded_at.desc()).all()

    return [CSAMetadataResponse.from_orm(csa) for csa in csas]


# Get CSA
@app.get("/api/v1/csa/{csa_id}", response_model=CSAMetadataResponse)
async def get_csa(csa_id: int, db: Session = Depends(get_db)):
    """Get CSA metadata"""
    csa = db.query(CSARecord).filter(CSARecord.id == csa_id).first()
    if not csa:
        raise HTTPException(status_code=404, detail="CSA not found")

    return CSAMetadataResponse.from_orm(csa)


# Download CSA
@app.get("/api/v1/csa/{csa_id}/download")
async def download_csa(csa_id: int, db: Session = Depends(get_db)):
    """Download CSA artifact"""
    csa = db.query(CSARecord).filter(CSARecord.id == csa_id).first()
    if not csa:
        raise HTTPException(status_code=404, detail="CSA not found")

    file_path = Path(csa.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(file_path, filename=file_path.name)


# Deploy CSA
@app.post("/api/v1/deployment/deploy")
async def deploy_csa(request: DeploymentRequest, db: Session = Depends(get_db)):
    """
    Deploy CSA to a site

    Args:
        request: Deployment request

    Returns:
        Deployment record
    """
    csa = db.query(CSARecord).filter(CSARecord.id == request.csa_id).first()
    if not csa:
        raise HTTPException(status_code=404, detail="CSA not found")

    # Create deployment record
    deployment = DeploymentRecord(
        csa_id=request.csa_id,
        site_id=request.site_id,
        deployed_by=request.deployed_by,
        status="deployed",
    )

    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    logger.info(f"Deployed CSA {request.csa_id} to site {request.site_id}")

    return {"deployment_id": deployment.id, "status": "deployed"}


# Rollback
@app.post("/api/v1/deployment/rollback")
async def rollback_deployment(request: RollbackRequest, db: Session = Depends(get_db)):
    """
    Rollback to a previous CSA version

    Args:
        request: Rollback request

    Returns:
        New deployment record
    """
    # Verify target CSA exists
    target_csa = db.query(CSARecord).filter(CSARecord.id == request.target_csa_id).first()
    if not target_csa:
        raise HTTPException(status_code=404, detail="Target CSA not found")

    # Get current deployment
    current_deployment = (
        db.query(DeploymentRecord)
        .filter(DeploymentRecord.site_id == request.site_id)
        .filter(DeploymentRecord.status == "deployed")
        .order_by(DeploymentRecord.deployed_at.desc())
        .first()
    )

    if current_deployment:
        # Mark as rolled back
        current_deployment.status = "rolled_back"
        current_deployment.rolled_back_at = datetime.utcnow()

    # Create new deployment
    deployment = DeploymentRecord(
        csa_id=request.target_csa_id,
        site_id=request.site_id,
        deployed_by="rollback_system",
        status="deployed",
    )

    db.add(deployment)
    db.commit()
    db.refresh(deployment)

    logger.info(f"Rolled back site {request.site_id} to CSA {request.target_csa_id}")

    return {"deployment_id": deployment.id, "status": "rolled_back"}


# Get deployment history
@app.get("/api/v1/deployment/history/{site_id}")
async def get_deployment_history(site_id: str, db: Session = Depends(get_db)):
    """Get deployment history for a site"""
    deployments = (
        db.query(DeploymentRecord)
        .filter(DeploymentRecord.site_id == site_id)
        .order_by(DeploymentRecord.deployed_at.desc())
        .all()
    )

    return [
        {
            "deployment_id": d.id,
            "csa_id": d.csa_id,
            "deployed_at": d.deployed_at,
            "status": d.status,
        }
        for d in deployments
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
