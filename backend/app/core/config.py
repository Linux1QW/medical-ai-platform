import json
import logging
import os
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_SECRET_KEY = "change-this-to-a-secure-random-string"


class Settings(BaseSettings):
    PROJECT_NAME: str = "医学问诊评估平台"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"  # development | production

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS: List[str] = ["Content-Type", "Authorization", "X-Request-ID"]

    # JWT
    SECRET_KEY: str = "change-this-to-a-secure-random-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_TOKEN_BLACKLIST_ENABLED: bool = True

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

    # ── Metadata 预过滤（按疾病/关键词缩小候选集，降噪提精度）──
    # disease_tags 等以 JSON 字符串存储，ChromaDB where 无法子串匹配，
    # 故改用 where_document={"$contains": ...} 对文档内容做子串过滤；
    # 命中不足时自动回退无过滤查询（见 medical_store.search / retriever）。
    ENABLE_METADATA_FILTER: bool = False    # 是否启用检索时的 metadata 预过滤
    METADATA_FILTER_MIN_RESULTS: int = 3    # 过滤结果少于该值时回退无过滤查询

    # ── A. 结果多样性重排（来源配额，避免 top-k 集中于单一来源）──
    ENABLE_DIVERSITY_RERANK: bool = False   # 粗排后、精排前是否施加来源多样性约束
    MAX_CHUNKS_PER_SOURCE: int = 2          # 同一 source 进入精排候选的最大条数

    # ── C. 融合 / 重排权重外置（便于基于评估集调参）──
    # 三路 RRF 权重 [BM25, Dense, Sparse]；两路时取前两项归一化
    RRF_WEIGHT_BM25: float = 0.30
    RRF_WEIGHT_DENSE: float = 0.45
    RRF_WEIGHT_SPARSE: float = 0.25
    # 两阶段重排最终打分权重（relevance/completeness/authority/freshness）
    RERANK_W_RELEVANCE: float = 0.4
    RERANK_W_COMPLETENESS: float = 0.3
    RERANK_W_AUTHORITY: float = 0.2
    RERANK_W_FRESHNESS: float = 0.1

    # ── B. Small-to-Big 上下文扩展（命中小块→拼相邻块喂 LLM）──
    ENABLE_CONTEXT_EXPANSION: bool = False  # 是否启用邻居块上下文拼接
    CONTEXT_EXPANSION_WINDOW: int = 1       # 向前/向后各拉取的邻居块数

    # ── D. 抽取式上下文压缩（喂 LLM 前句级降噪）──
    ENABLE_CONTEXT_COMPRESSION: bool = False  # 是否对长证据做句级抽取
    COMPRESSION_MIN_CHARS: int = 400          # 证据短于该长度不压缩
    COMPRESSION_TOP_SENTENCES: int = 5        # 每条证据最多保留的句数
    COMPRESSION_MIN_SENT_SCORE: float = 0.2   # 句子与查询相似度低于该值则丢弃

    # ── OCR（扫描版/图片型 PDF 文字识别兜底）──
    # 复用 Qwen-VL 多模态能力，仅在页面文本过少时触发，未配置 API Key 时自动降级
    ENABLE_OCR: bool = False                # 是否对低文本页启用 Qwen-VL OCR 兜底
    OCR_MODEL: str = "qwen-vl-ocr"          # 通义千问多模态 OCR 模型
    OCR_MIN_TEXT_THRESHOLD: int = 50        # 页面文本低于该字符数视为扫描/图片页
    OCR_RENDER_DPI: int = 200               # 渲染 PDF 页为图片的 DPI
    OCR_MAX_CONCURRENCY: int = 4            # OCR API 并发上限
    OCR_TIMEOUT_SECONDS: int = 60           # 单页 OCR 请求超时（秒）

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

    # Retrieval Cache
    RETRIEVAL_CACHE_ENABLED: bool = True
    RETRIEVAL_CACHE_TTL: int = 86400            # 24 hours
    RETRIEVAL_CACHE_MAX_SIZE: int = 5000

    # LLM 响应缓存
    LLM_CACHE_ENABLED: bool = True                # 是否启用 LLM 响应缓存
    LLM_CACHE_TTL: int = 86400                    # 缓存过期时间（秒），24小时
    LLM_CACHE_SIMILARITY_THRESHOLD: float = 0.95  # 语义相似度阈值（保留，当前使用精确哈希匹配）
    LLM_CACHE_MAX_SIZE: int = 10000               # 最大缓存条目数

    # Testing mode (skip real connections in tests)
    TESTING: bool = False                         # 测试模式，避免连接真实服务

    # LLM 建议生成
    ENABLE_LLM_SUGGESTION: bool = True            # 启用 LLM 建议生成（false 时回退规则建议）

    # 审计日志
    AUDIT_LOG_ENABLED: bool = True

    # 数据留存策略
    AUDIT_LOG_RETENTION_DAYS: int = 90        # 审计日志保留天数
    EVALUATION_RUN_RETENTION_DAYS: int = 180  # 评估运行记录保留天数

    # Token 用量管控
    TOKEN_DAILY_LIMIT: int = 1_000_000       # 每日 Token 上限
    COST_PER_1K_TOKENS: float = 0.02         # 每千 Token 成本（元），用于估算

    # 告警配置
    ALERT_WEBHOOK_URL: str = ""           # 钉钉/企微 Webhook URL
    ALERT_WEBHOOK_TYPE: str = "dingtalk"  # dingtalk | wecom
    LLM_ERROR_RATE_THRESHOLD: float = 0.1  # 10% 错误率告警

    # Celery 异步任务队列
    CELERY_BROKER_URL: str = "redis://localhost:6379/4"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/5"

    # 模型路由（降级策略）
    MODEL_CRITICAL: str = "qwen-max"        # 评估智能体（核心）
    MODEL_STANDARD: str = "qwen-plus"       # 人文关怀评估
    MODEL_LIGHTWEIGHT: str = "qwen-turbo"   # 输入清洗/格式化

    # 跨 Provider LLM Fallback
    LLM_PROVIDERS: str = "[]"               # JSON 格式，支持多个 Provider 配置
    LLM_CIRCUIT_BREAKER_THRESHOLD: int = 3  # 连续失败 N 次后切换备用 Provider

    # ── 通用 LLM 接入（Provider / 框架无关）──
    # 通过 ProviderAdapter 抽象层解耦具体厂商；下列通用配置为空时回退 QWEN_* ，
    # 从而在不破坏现有阿里云百炼配置的前提下支持接入任意 OpenAI 兼容服务。
    LLM_PROVIDER_TYPE: str = "openai_compatible"  # 适配器类型，对应 ProviderAdapter 注册表
    LLM_API_KEY: str = ""        # 为空时回退 QWEN_API_KEY
    LLM_API_BASE_URL: str = ""   # 为空时回退 QWEN_API_BASE_URL
    LLM_MODEL: str = ""          # 为空时回退 QWEN_MODEL

    # Prompt 版本管理
    # JSON 映射 {"<prompt_key>": "<version>"}，覆盖 manifest.json 中的活跃版本，
    # 便于灰度 / A-B 对比而无需改动 Prompt 文件；空对象表示全部使用 manifest.active。
    PROMPT_ACTIVE_VERSIONS: str = "{}"

    # 日志配置
    LOG_LEVEL: str = "INFO"          # 日志级别：DEBUG | INFO | WARNING | ERROR
    LOG_FORMAT: str = "json"         # 日志格式：json | text

    # Langfuse 链路追踪
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = False

    # BGE-M3 双表示配置
    BGE_M3_ENABLED: bool = False          # 默认关闭，需要时通过环境变量开启
    BGE_M3_MODEL_PATH: str = "BAAI/bge-m3"  # 模型路径或 HuggingFace ID
    BGE_M3_USE_FP16: bool = False         # GPU 环境开启 FP16 量化
    BGE_M3_QUERY_INSTRUCTION: str = "为这个医学查询生成检索表示："

    # 数据库连接池配置
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600     # 连接回收时间（秒）
    DB_POOL_TIMEOUT: int = 30       # 获取连接超时（秒）

    def get_llm_providers(self) -> list[dict]:
        """解析 LLM_PROVIDERS JSON 配置"""
        try:
            providers = json.loads(self.LLM_PROVIDERS)
            if isinstance(providers, list) and providers:
                return providers
        except (json.JSONDecodeError, TypeError):
            pass
        # 默认返回当前单 Provider 配置（通用 LLM_* 优先，回退 QWEN_*）
        return [{
            "name": "default",
            "type": self.LLM_PROVIDER_TYPE,
            "api_key": self.llm_api_key,
            "base_url": self.llm_api_base_url,
            "model": self.llm_model,
        }]

    @property
    def llm_api_key(self) -> str:
        """通用 LLM API Key（未配置时回退 QWEN_API_KEY）。"""
        return self.LLM_API_KEY or self.QWEN_API_KEY

    @property
    def llm_api_base_url(self) -> str:
        """通用 LLM Base URL（未配置时回退 QWEN_API_BASE_URL）。"""
        return self.LLM_API_BASE_URL or self.QWEN_API_BASE_URL

    @property
    def llm_model(self) -> str:
        """通用 LLM 模型（未配置时回退 QWEN_MODEL）。"""
        return self.LLM_MODEL or self.QWEN_MODEL

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def check_security(self) -> None:
        """启动时安全检查配置"""
        if self.TESTING:
            return
        if self.SECRET_KEY == _DEFAULT_SECRET_KEY:
            if self.ENVIRONMENT == "production":
                raise RuntimeError(
                    "SECRET_KEY 未设置！生产环境必须通过环境变量 SECRET_KEY 配置安全密钥。"
                )
            logger.warning(
                "SECURITY WARNING: SECRET_KEY 仍为默认值！"
                "请在生产环境中设置安全的随机密钥（环境变量 SECRET_KEY）。"
            )


settings = Settings()
