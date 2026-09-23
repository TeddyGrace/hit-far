"""Object storage behind a small interface.

`S3Storage` targets any S3-compatible service (Railway Buckets, Cloudflare R2, AWS S3).
`LocalStorage` keeps files on disk and issues signed URLs served by `routers/local_storage.py`,
so the whole app runs locally and in tests without a bucket.
"""

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.config import get_settings


class Storage(Protocol):
    def presign_put(self, key: str, content_type: str, expires_s: int = 3600) -> str: ...
    def presign_get(self, key: str, expires_s: int = 3600) -> str: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    def upload_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None: ...
    def download_file(self, key: str, path: Path) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def copy(self, src_key: str, dst_key: str) -> None: ...
    def delete(self, key: str) -> None: ...


class S3Storage:
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        s = get_settings()
        if not s.s3_bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET (or BUCKET_NAME)")
        self.bucket = s.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            aws_access_key_id=s.s3_access_key_id,
            aws_secret_access_key=s.s3_secret_access_key,
            region_name=s.s3_region,
            config=Config(signature_version="s3v4", s3={"addressing_style": s.s3_addressing_style}),
        )

    def presign_put(self, key: str, content_type: str, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires_s,
        )

    def presign_get(self, key: str, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def upload_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None:
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})

    def download_file(self, key: str, path: Path) -> None:
        self.client.download_file(self.bucket, key, str(path))

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def copy(self, src_key: str, dst_key: str) -> None:
        self.client.copy({"Bucket": self.bucket, "Key": src_key}, self.bucket, dst_key)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def set_cors(self, origins: list[str]) -> None:
        self.client.put_bucket_cors(
            Bucket=self.bucket,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedOrigins": origins,
                        "AllowedMethods": ["GET", "PUT", "HEAD"],
                        "AllowedHeaders": ["*"],
                        "ExposeHeaders": ["ETag"],
                        "MaxAgeSeconds": 3600,
                    }
                ]
            },
        )


_LOCAL_SALT = "local-storage"


class LocalStorage:
    def __init__(self) -> None:
        s = get_settings()
        self.root = s.local_storage_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.signer = URLSafeTimedSerializer(s.secret_key, salt=_LOCAL_SALT)

    def path_for(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("invalid key")
        return p

    def _signed_url(self, key: str, method: str) -> str:
        return f"/api/storage/local/{self.signer.dumps({'k': key, 'm': method})}"

    def verify(self, token: str, method: str, max_age: int = 3600) -> str:
        try:
            data = self.signer.loads(token, max_age=max_age)
        except (BadSignature, SignatureExpired) as e:
            raise PermissionError("invalid or expired storage token") from e
        if data.get("m") != method:
            raise PermissionError("wrong method for token")
        return data["k"]

    def presign_put(self, key: str, content_type: str, expires_s: int = 3600) -> str:
        return self._signed_url(key, "PUT")

    def presign_get(self, key: str, expires_s: int = 3600) -> str:
        return self._signed_url(key, "GET")

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self.path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def upload_file(self, key: str, path: Path, content_type: str = "application/octet-stream") -> None:
        self.put_bytes(key, Path(path).read_bytes())

    def download_file(self, key: str, path: Path) -> None:
        Path(path).write_bytes(self.path_for(key).read_bytes())

    def get_bytes(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def copy(self, src_key: str, dst_key: str) -> None:
        self.put_bytes(dst_key, self.get_bytes(src_key))

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)


@lru_cache
def get_storage() -> Storage:
    if get_settings().storage_backend == "s3":
        return S3Storage()
    return LocalStorage()
