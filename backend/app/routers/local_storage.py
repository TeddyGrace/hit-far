"""Serves signed URLs for STORAGE_BACKEND=local (dev/tests). Stands in for presigned S3 URLs."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.storage import LocalStorage, get_storage

router = APIRouter(prefix="/api/storage/local", tags=["storage"], include_in_schema=False)


def _local() -> LocalStorage:
    st = get_storage()
    if not isinstance(st, LocalStorage):
        raise HTTPException(404)
    return st


@router.put("/{token}")
async def put_object(token: str, request: Request):
    st = _local()
    try:
        key = st.verify(token, "PUT")
    except PermissionError as e:
        raise HTTPException(403, str(e)) from None
    p = st.path_for(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    return {"ok": True}


@router.get("/{token}")
def get_object(token: str):
    st = _local()
    try:
        key = st.verify(token, "GET", max_age=6 * 3600)
    except PermissionError as e:
        raise HTTPException(403, str(e)) from None
    p = st.path_for(key)
    if not p.is_file():
        raise HTTPException(404)
    media = "video/mp4" if p.suffix == ".mp4" else None
    return FileResponse(p, media_type=media)  # supports Range requests (video seeking)
