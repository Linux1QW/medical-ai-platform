import logging
import os

from pydantic_settings import BaseSettings
from typing import List

logger = logging.getLogger(__name__)

_DEFAULT_SECRET_KEY = "change-this-to-a-secure-random-string"


class Settings(BaseSettings):
    PROJECT_NAME: str = "医学问诊评估平台"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # JWT
    SECRET_KEY: str = "change-this-to-a-secure-random-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "medical_ai"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # 阿里云百炼平台 Qwen API — 优先从系统环境变量 DASHSCOPE_API_KEY 读取
    QWEN_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")
    QWEN_API_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen3.7-max"

    # Rerank 模型
    RERANK_MODEL: str = "gte-rerank"

    # LLM 并发控制
    # 全局同时允许的最大 LLM API 调用数（防止触发 API 限流 429）
    LLM_MAX_CONCURRENT: int = 10
    # 等待信号量超时（秒），超时后抛出异常避免无限排队
    LLM_SEMAPHORE_TIMEOUT: int = 60

    # RAG 索引版本
    ACTIVE_INDEX_VERSION: str = "rag-v1"    # 当前活跃版本

    # LangGraph 编排
    LANGGRAPH_ENABLED: bool = True          # Feature Flag，False 时回退旧编排
    LANGGRAPH_SHADOW_MODE: bool = False     # 影子模式（同时运行新旧编排，只记录不返回新结果）
    LANGGRAPH_GRAPH_VERSION: str = "evaluation-graph-v1"
    LANGGRAPH_CHECKPOINT_DB: str = "backend/data/langgraph_checkpoints.sqlite3"  # 已弃用，保留作为回退
    LANGGRAPH_CHECKPOINT_TTL_HOURS: int = 24

    # Redis Checkpoint
    REDIS_CHECKPOINT_URL: str = "redis://localhost:6379/1"  # 使用 db=1 避免与应用缓存冲突
    REDIS_CHECKPOINT_TTL: int = 86400  # 24小时过期（秒）

    # Function Call / Tool Use
    ENABLE_TOOL_USE: bool = True
    TOOL_USE_MODEL: str = "qwen-max"
    TOOL_USE_MAX_ROUNDS: int = 4
    TOOL_USE_MAX_CALLS: int = 8
    TOOL_USE_TIMEOUT_SECONDS: int = 30
    TOOL_USE_MAX_RESULT_CHARS: int = 6000
    KNOWLEDGE_TOOL_MAX_RAG_CALLS: int = 3
    KNOWLEDGE_TOOL_MAX_MQE_CALLS: int = 2
    KNOWLEDGE_TOOL_MAX_HYDE_CALLS: int = 1
    TOOL_USE_FALLBACK_TO_LEGACY: bool = True

    # ReAct 模式配置
    ENABLE_REACT_KNOWLEDGE: bool = True           # Knowledge Agent 启用 ReAct 模式
    ENABLE_REACT_REFLECTION: bool = True          # Reflection Agent 启用 ReAct 模式
    REACT_MAX_STEPS: int = 6                      # ReAct 最大推理步数（Thought→Action→Observation）
    REFLECTION_CONSISTENCY_THRESHOLD: float = 0.3 # 评分一致性偏差阈值（超过则标记矛盾）
    REFLECTION_EVIDENCE_MIN_SCORE: float = 60.0   # 反思时认为证据充足的最低分数

    # LLM 响应缓存
    LLM_CACHE_ENABLED: bool = True                # 是否启用 LLM 响应缓存
    LLM_CACHE_TTL: int = 86400                    # 缓存过期时间（秒），24小时
    LLM_CACHE_SIMILARITY_THRESHOLD: float = 0.95  # 语义相似度阈值（保留，当前使用精确哈希匹配）
    LLM_CACHE_MAX_SIZE: int = 10000               # 最大缓存条目数

    # LLM 建议生成
    ENABLE_LLM_SUGGESTION: bool = True            # 启用 LLM 建议生成（false 时回退规则建议）

    # 审计日志
    AUDIT_LOG_ENABLED: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def check_security(self) -> None:
        """启动时安全检查配置"""
        if self.SECRET_KEY == _DEFAULT_SECRET_KEY:
            logger.warning(
                "SECURITY WARNING: SECRET_KEY 仍为默认值！"
                "请在生产环境中设置安全的随机密钥（环境变量 SECRET_KEY）。"
            )


settings = Settings()
