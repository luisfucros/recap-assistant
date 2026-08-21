"""Object storage over the S3 API — one implementation, two deployments.

The same client talks to MinIO locally (via ``S3_ENDPOINT_URL``) and AWS S3 in
production (endpoint unset ⇒ the AWS default). Credentials may be omitted so the
AWS default credential chain (IAM role, env, profile) is used.
"""

import aioboto3

from shared.core.config import Settings


class S3StorageProvider:
    """S3-compatible object storage via ``aioboto3``.

    A fresh client is opened per call: ``aioboto3`` clients are async context
    managers and are not safe to share across the process, so we scope one to
    each operation rather than hold a long-lived connection.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
        session: aioboto3.Session | None = None,
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint_url
        self._access = access_key
        self._secret = secret_key
        self._region = region
        self._session = session or aioboto3.Session()

    def _client(self):  # noqa: ANN202 — aioboto3's client context manager type is dynamic
        return self._session.client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access,
            aws_secret_access_key=self._secret,
            region_name=self._region,
        )

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        async with self._client() as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    async def get(self, key: str) -> bytes:
        async with self._client() as s3:
            resp = await s3.get_object(Bucket=self._bucket, Key=key)
            async with resp["Body"] as stream:
                return await stream.read()

    async def delete(self, key: str) -> None:
        async with self._client() as s3:
            await s3.delete_object(Bucket=self._bucket, Key=key)


def build_storage_provider(settings: Settings) -> S3StorageProvider:
    """Construct the S3/MinIO storage provider from settings."""
    return S3StorageProvider(
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        access_key=(
            settings.s3_access_key_id.get_secret_value() if settings.s3_access_key_id else None
        ),
        secret_key=(
            settings.s3_secret_access_key.get_secret_value()
            if settings.s3_secret_access_key
            else None
        ),
        region=settings.s3_region,
    )
