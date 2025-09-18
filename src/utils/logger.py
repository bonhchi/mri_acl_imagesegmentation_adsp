# src/utils/logger.py
import logging, json, os, sys, socket, time, uuid
from logging.handlers import TimedRotatingFileHandler

RUN_ID = os.environ.get("RUN_ID") or time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
HOST = socket.gethostname()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": RUN_ID,
            "host": HOST,
            # === Thông tin process/thread & vị trí code ===
            "pid": record.process,
            "processName": record.processName,
            "tid": record.thread,
            "threadName": record.threadName,
            "module": record.module,
            "func": record.funcName,
            "lineno": record.lineno,
            "pathname": record.pathname,
        }
        # gộp extra dict nếu có
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        # thêm fields từ LogRecord (nếu truyền bằng logger.info("...", extra={...}))
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in base and k not in ("args", "msg", "message", "exc_text", "exc_info"):
                if isinstance(v, (str, int, float, bool, dict, list, type(None))):
                    base[k] = v
        return json.dumps(base, ensure_ascii=False)

def _make_handler(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    h = TimedRotatingFileHandler(log_path, when="midnight", backupCount=7, encoding="utf-8")
    h.setFormatter(JsonFormatter())
    h.setLevel(logging.INFO)
    return h

def get_logger(name: str, log_file: str | None = None, console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # tránh add trùng handler khi gọi nhiều lần
    if logger.handlers: 
        return logger
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(JsonFormatter())
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
    if log_file:
        logger.addHandler(_make_handler(log_file))
    return logger

def set_run_id(run_id: str):
    global RUN_ID
    RUN_ID = run_id
