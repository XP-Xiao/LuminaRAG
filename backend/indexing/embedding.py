"""文本向量化服务 - 通过 SiliconFlow embedding API（BAAI/bge-m3，1024 维）"""
import os

import requests


class EmbeddingService:
    """文本向量化服务 - 稠密向量（SiliconFlow API 调用，1024 维 bge-m3）"""

    def __init__(self):
        self._api_key = os.getenv("EMBEDDING_API_KEY", "")
        self._base_url = os.getenv(
            "EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1/embeddings"
        )
        self._model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = requests.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self._model, "input": texts},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            items.sort(key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in items]
        except Exception as e:
            raise Exception(f"嵌入 API 调用失败: {str(e)}") from e


# 全进程唯一实例
embedding_service = EmbeddingService()
