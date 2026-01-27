from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class UploadConfig:
    provider: str = "tos"
    bucket: Optional[str] = None
    prefix: str = ""
    public: bool = False
    expires: int = 3600


class BaseUploader:
    def upload_file(self, local_path: str, key: Optional[str] = None) -> str:
        raise NotImplementedError

    def upload_files(self, local_paths: List[str]) -> List[str]:
        return [self.upload_file(p) for p in local_paths]


class TOSUploader(BaseUploader):
    """Upload files to Volc TOS. Requires ve-tos-python-sdk (import as `tos`)."""

    def __init__(self, cfg: UploadConfig) -> None:
        self.cfg = cfg
        self.bucket = cfg.bucket or os.getenv("TOS_BUCKET") or os.getenv("MLLM_UPLOAD_BUCKET")
        if not self.bucket:
            raise EnvironmentError("TOSUploader requires bucket (set TOS_BUCKET/MLLM_UPLOAD_BUCKET or cfg.bucket)")
        self._client = None

    def _make_client(self):
        if self._client is not None:
            return self._client
        try:
            import tos  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("ve-tos-python-sdk is required for TOSUploader. `pip install ve-tos-python-sdk`.") from exc

        ak = os.getenv("TOS_AK")
        sk = os.getenv("TOS_SK")
        endpoint = os.getenv("TOS_ENDPOINT")
        region = os.getenv("TOS_REGION")
        if not ak or not sk or not endpoint or not region:
            raise EnvironmentError("Please set TOS_AK, TOS_SK, TOS_ENDPOINT, TOS_REGION for TOS upload")
        self._client = tos.TosClientV2(ak, sk, endpoint, region)
        return self._client

    def _public_url(self, key: str) -> str:
        endpoint = os.getenv("TOS_ENDPOINT") or "tos-cn-beijing.volces.com"
        return f"https://{self.bucket}.{endpoint}/{key}"

    def upload_file(self, local_path: str, key: Optional[str] = None) -> str:
        cli = self._make_client()
        local = Path(local_path)
        if not key:
            key = (Path(self.cfg.prefix) / local.name).as_posix() if self.cfg.prefix else local.name
        with open(local, "rb") as f:
            cli.put_object(self.bucket, key, content=f.read())
        if self.cfg.public:
            return self._public_url(key)
        resp = cli.pre_signed_url(  # type: ignore
            __import__("tos").HttpMethodType.Http_Method_Get,
            self.bucket,
            key,
            int(self.cfg.expires),
        )
        return resp.signed_url


def build_uploader(cfg: UploadConfig | None) -> Optional[BaseUploader]:
    if not cfg:
        provider = (os.getenv("MLLM_UPLOAD_PROVIDER") or os.getenv("UPLOAD_PROVIDER") or "").lower()
        if not provider:
            return None
        cfg = UploadConfig(
            provider=provider,
            bucket=os.getenv("MLLM_UPLOAD_BUCKET") or os.getenv("TOS_BUCKET"),
            prefix=os.getenv("MLLM_UPLOAD_PREFIX", ""),
            public=bool(int(os.getenv("MLLM_UPLOAD_PUBLIC", os.getenv("UPLOAD_PUBLIC", "0")))),
            expires=int(os.getenv("MLLM_UPLOAD_EXPIRES", os.getenv("UPLOAD_EXPIRES", "3600"))),
        )
    provider = (cfg.provider or "").lower()
    if provider == "tos":
        return TOSUploader(cfg)
    raise ValueError(f"Unsupported upload provider: {cfg.provider}")

