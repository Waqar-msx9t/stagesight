"""Where results live: a local folder on a laptop, an S3 bucket on AWS. Same keys either way.

STORE=local:<folder>   (default local:var/bucket)
STORE=s3:<bucket>
"""
from __future__ import annotations

import json
import os


class LocalStore:
    def __init__(self, root: str):
        self.root = root

    def _path(self, key: str) -> str:
        return os.path.join(self.root, *key.split("/"))

    def get(self, key: str) -> bytes | None:
        try:
            with open(self._path(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


class S3Store:
    def __init__(self, bucket: str):
        import boto3  # only needed on AWS
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                           CacheControl="no-cache")


def from_env() -> LocalStore | S3Store:
    spec = os.environ.get("STORE", "local:var/bucket")
    kind, _, where = spec.partition(":")
    return S3Store(where) if kind == "s3" else LocalStore(where)


def get_json(store, key: str, default=None):
    data = store.get(key)
    return json.loads(data) if data is not None else default


def put_json(store, key: str, obj) -> None:
    store.put(key, json.dumps(obj).encode(), "application/json")
