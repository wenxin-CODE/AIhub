"""StarOps 全局配置。

配置来源优先级（高 → 低）：

1. **进程环境变量** —— CI / 容器 / shell export，最权威，`.env` 不得覆盖
2. **项目根 `.env` 文件** —— 本地开发用；已存在的环境变量不会被它覆盖
3. **代码内默认值** —— 保证零配置也能跑起来

敏感项（API Key）不得硬编码：`.env` 已被 `.gitignore` 排除，
`.env.example` 只保留占位符。

诊断用法：

```bash
cd starops/backend && ../.venv/Scripts/python.exe config.py
```
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ENV_FILE = BASE_DIR / ".env"


# ---------------------------------------------------------------------------
# .env 解析与注入
# ---------------------------------------------------------------------------

# 从 .env 文件解析出的全部键值
_DOTENV_PARSED: dict[str, str] = {}
# 实际由 .env 注入 os.environ 的键（其余键原本就存在于进程环境中）
_DOTENV_INJECTED: set[str] = set()


def _strip_quotes(value: str) -> str:
    """去掉成对包裹的单/双引号。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    """剥离未加引号值的行尾注释。

    只有 ``#`` 前有空白才算注释起点——否则 URL 里的 ``#fragment``
    会被误吃掉（``https://x/y#z`` 保持原样）。
    引号包裹的值不做处理，其内部的 ``#`` 是内容。
    """
    if value[:1] in "\"'":
        return value
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1] in " \t":
            return value[:i].rstrip()
    return value


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """把 `.env` 文件里的键值注入 ``os.environ``，返回解析结果。

    规则：

    - **真实环境变量优先**：已存在的键不会被 `.env` 覆盖
      （与 python-dotenv 的 ``override=False`` 语义一致）
    - 支持 ``export KEY=VALUE``、空行、``#`` 注释、单/双引号包裹
    - 文件不存在时静默跳过——生产环境用环境变量注入即可，不需要 `.env`
    """
    env_path = path or ENV_FILE
    parsed: dict[str, str] = {}
    injected: set[str] = set()

    if env_path.is_file():
        try:
            content = env_path.read_text(encoding="utf-8")
        except OSError:
            content = ""
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            value = _strip_quotes(_strip_inline_comment(value.strip()))
            parsed[key] = value
            if key not in os.environ:
                os.environ[key] = parsed[key]
                injected.add(key)

    global _DOTENV_PARSED, _DOTENV_INJECTED
    _DOTENV_PARSED = parsed
    _DOTENV_INJECTED = injected
    return parsed


def _resolve_path(raw: str) -> Path:
    """把配置里的路径解析成绝对路径。

    相对路径按**项目根 BASE_DIR** 解析，而不是当前工作目录——
    否则 ``cd backend && uvicorn`` 与 ``cd starops && uvicorn``
    会落到不同的数据库和目录，表现为"数据莫名消失"。
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()


# 必须在 Settings 类定义之前完成加载：类属性在导入时求值。
load_dotenv()


class Settings:
    """应用配置。

    路径类字段统一走 :func:`_resolve_path`；LLM 类字段做成 property，
    使得在导入之后被修改（测试注入、热更新）也能读到当前值，
    而不是导入那一刻的快照。
    """

    app_name: str = "StarOps"
    api_prefix: str = "/api"
    debug: bool = os.getenv("STAROPS_DEBUG", "0") == "1"

    # ---- 数据库 ----
    @property
    def db_path(self) -> Path:
        return _resolve_path(os.getenv("STAROPS_DB_PATH", "data/starops.db"))

    @property
    def db_echo(self) -> bool:
        return os.getenv("STAROPS_DB_ECHO", "0") == "1"

    # ---- Agent ----
    @property
    def checkpoint_dir(self) -> Path:
        return _resolve_path(os.getenv("STAROPS_CHECKPOINT_DIR", "data/checkpoints"))

    @property
    def max_iterations(self) -> int:
        return int(os.getenv("STAROPS_MAX_ITERATIONS", "12"))

    # ---- 影像 ----
    @property
    def image_dir(self) -> Path:
        return _resolve_path(os.getenv("STAROPS_IMAGE_DIR", "data/images"))

    # ---- 审计 ----
    audit_enabled: bool = os.getenv("STAROPS_AUDIT_ENABLED", "1") == "1"
    # 记录模型版本 / 提示词版本 / 工具版本，便于复现
    trace_model_version: str = os.getenv("STAROPS_TRACE_MODEL_VERSION", "v1")
    trace_prompt_version: str = os.getenv("STAROPS_TRACE_PROMPT_VERSION", "p1")

    # ---- LLM（property：运行时可读当前值）----
    @property
    def llm_provider(self) -> str:
        return os.getenv("STAROPS_LLM_PROVIDER", "deepseek")

    @property
    def llm_api_key(self) -> str:
        """API Key。

        ``STAROPS_LLM_API_KEY`` 优先于 ``DEEPSEEK_API_KEY``——
        前者与 provider 解耦，切换到任意 OpenAI 兼容端点时不必改变量名。
        """
        return (
            os.getenv("STAROPS_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        ).strip()

    @property
    def llm_base_url(self) -> str:
        return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    @property
    def llm_model(self) -> str:
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    @property
    def llm_timeout(self) -> int:
        return int(os.getenv("STAROPS_LLM_TIMEOUT", "60"))

    @property
    def llm_max_retries(self) -> int:
        return int(os.getenv("STAROPS_LLM_MAX_RETRIES", "2"))

    @property
    def llm_available(self) -> bool:
        """未配置 Key 时降级为 mock，保证离线可演示。"""
        return bool(self.llm_api_key)

    @property
    def llm_api_key_source(self) -> str:
        """Key 的来源标签，仅供诊断——**不回显 Key 内容**。

        取值：``unset`` / ``process-env`` / ``dotenv`` / ``dotenv+process-env``
        """
        for name in ("STAROPS_LLM_API_KEY", "DEEPSEEK_API_KEY"):
            raw = os.getenv(name, "").strip()
            if not raw:
                continue
            if name in _DOTENV_PARSED:
                if name in _DOTENV_INJECTED:
                    return "dotenv"
                return "dotenv+process-env"
            return "process-env"
        return "unset"

    @property
    def env_file(self) -> Path:
        return ENV_FILE

    def diagnostics(self) -> dict:
        """配置诊断快照：脱敏、可安全打日志或返回前端。"""
        key = self.llm_api_key
        return {
            "env_file": self.env_file.as_posix(),
            "env_file_exists": self.env_file.is_file(),
            "env_file_parsed_keys": sorted(_DOTENV_PARSED.keys()),
            "llm_api_key_set": bool(key),
            "llm_api_key_source": self.llm_api_key_source,
            "llm_api_key_masked": (
                f"{key[:6]}...{key[-4:]}(len={len(key)})" if len(key) > 12 else "***"
            ),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_available": self.llm_available,
            "db_path": self.db_path.as_posix(),
            "checkpoint_dir": self.checkpoint_dir.as_posix(),
            "image_dir": self.image_dir.as_posix(),
        }

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()


if __name__ == "__main__":
    import json

    print(json.dumps(settings.diagnostics(), ensure_ascii=False, indent=2))
