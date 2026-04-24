import asyncio
import json
from typing import Any

from aiohttp import web

from src.plugins.base import Plugin
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Middleware ────────────────────────────────────────────────────────────────


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """CORS 中间件：注入跨域头，处理 OPTIONS 预检请求。"""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@web.middleware
async def error_middleware(request: web.Request, handler):
    """错误处理中间件：捕获未预期异常返回 500，处理 404。"""
    try:
        response = await handler(request)
        # aiohttp 对未匹配路由返回 status >= 400 的普通 Response（非异常）
        if response.status == 404:
            return web.json_response(
                {"success": False, "error": "未找到请求的资源"},
                status=404,
            )
        return response
    except web.HTTPException:
        raise
    except Exception as e:
        logger.error(f"API 处理异常: {e}", exc_info=True)
        return web.json_response(
            {"success": False, "error": "内部服务器错误"},
            status=500,
        )


@web.middleware
async def content_type_middleware(request: web.Request, handler):
    """POST Content-Type 验证中间件：要求 application/json。"""
    if request.method == "POST":
        content_type = request.content_type or ""
        if not content_type.startswith("application/json"):
            return web.json_response(
                {"success": False, "error": "Content-Type 必须为 application/json"},
                status=400,
            )
    return await handler(request)


# ── Plugin ───────────────────────────────────────────────────────────────────


class HttpApiPlugin(Plugin):
    """内嵌 HTTP API 服务插件，提供 REST API 和 SSE 事件流。"""

    name = "http_api"
    priority = 65

    def __init__(self) -> None:
        super().__init__()
        self.app = None
        self._host: str = "0.0.0.0"
        self._port: int = 8000
        self._web_app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._sse_queues: set[asyncio.Queue] = set()
        self._current_chat_message: str = ""
        self._current_emotion: str = "neutral"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def setup(self, app: Any) -> None:
        """读取配置，创建 aiohttp.web.Application，注册路由和中间件。"""
        self.app = app

        config = ConfigManager.get_instance()
        self._host = config.get_config("HTTP_API_OPTIONS.HOST", "0.0.0.0")
        self._port = int(config.get_config("HTTP_API_OPTIONS.PORT", 8000))

        self._web_app = web.Application(
            middlewares=[
                cors_middleware,
                error_middleware,
                content_type_middleware,
            ]
        )
        self._register_routes()

    async def start(self) -> None:
        """启动 aiohttp AppRunner + TCPSite，绑定端口。"""
        if not self._web_app:
            return
        try:
            self._runner = web.AppRunner(self._web_app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            self._started = True
            logger.info(f"HTTP API 服务已启动: http://{self._host}:{self._port}")
        except OSError as e:
            logger.error(f"HTTP API 启动失败（端口 {self._port} 可能被占用）: {e}")
            self._started = False

    async def stop(self) -> None:
        """关闭所有 SSE 连接。"""
        for queue in list(self._sse_queues):
            await queue.put(None)  # sentinel to break SSE loops
        self._sse_queues.clear()
        self._started = False

    async def shutdown(self) -> None:
        """停止 aiohttp 服务，释放端口资源。"""
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    # ── Route registration ────────────────────────────────────────────────

    def _register_routes(self) -> None:
        assert self._web_app is not None
        self._web_app.router.add_get("/api/status", self._handle_status)
        self._web_app.router.add_get("/api/events", self._handle_events)
        self._web_app.router.add_post(
            "/api/action/start-conversation", self._handle_start_conversation
        )
        self._web_app.router.add_post(
            "/api/action/abort-speaking", self._handle_abort_speaking
        )
        self._web_app.router.add_post(
            "/api/action/send-text", self._handle_send_text
        )
        self._web_app.router.add_post(
            "/api/action/start-listening", self._handle_start_listening
        )
        self._web_app.router.add_post(
            "/api/action/stop-listening", self._handle_stop_listening
        )

    # ── Route handlers (stubs – full implementation in Task 2 & 4) ────────

    async def _handle_status(self, request: web.Request) -> web.Response:
        snapshot = self.app.get_state_snapshot()
        snapshot["chat_message"] = self._current_chat_message
        snapshot["emotion"] = self._current_emotion
        return web.json_response({"success": True, "data": snapshot})

    async def _handle_events(self, request: web.Request) -> web.StreamResponse:
        # SSE – full implementation in Task 4
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)

        queue: asyncio.Queue = asyncio.Queue()
        self._sse_queues.add(queue)
        try:
            while True:
                payload = await queue.get()
                if payload is None:
                    break
                await response.write(payload.encode("utf-8"))
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._sse_queues.discard(queue)
        return response

    async def _handle_start_conversation(self, request: web.Request) -> web.Response:
        await self.app.start_auto_conversation()
        return web.json_response({"success": True, "data": None})

    async def _handle_abort_speaking(self, request: web.Request) -> web.Response:
        await self.app.abort_speaking(None)
        return web.json_response({"success": True, "data": None})

    async def _handle_send_text(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"success": False, "error": "无效的 JSON 请求体"},
                status=400,
            )
        text = body.get("text") if isinstance(body, dict) else None
        if not text or not isinstance(text, str) or not text.strip():
            return web.json_response(
                {"success": False, "error": "text 字段不能为空"},
                status=400,
            )
        ui_plugin = self.app.plugins.get_plugin("ui")
        if ui_plugin and hasattr(ui_plugin, "_send_text"):
            await ui_plugin._send_text(text)
        return web.json_response({"success": True, "data": None})

    async def _handle_start_listening(self, request: web.Request) -> web.Response:
        await self.app.start_listening_manual()
        return web.json_response({"success": True, "data": None})

    async def _handle_stop_listening(self, request: web.Request) -> web.Response:
        await self.app.stop_listening_manual()
        return web.json_response({"success": True, "data": None})

    # ── SSE broadcast ─────────────────────────────────────────────────────

    def _broadcast_sse(self, event: str, data: dict) -> None:
        """向所有 SSE 客户端队列写入事件。"""
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        for queue in list(self._sse_queues):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ── Event hooks ───────────────────────────────────────────────────────

    async def on_device_state_changed(self, state: Any) -> None:
        """将状态变化推送到所有 SSE 客户端。"""
        self._broadcast_sse("state_changed", {"device_state": str(state)})

    async def on_incoming_json(self, message: Any) -> None:
        """将文本/表情变化推送到所有 SSE 客户端。"""
        if not isinstance(message, dict):
            return
        msg_type = message.get("type")
        if msg_type in ("tts", "stt"):
            text = message.get("text", "")
            state = message.get("state", "")
            if text:
                role = "assistant" if msg_type == "tts" else "user"
                self._current_chat_message = text
                event_data = {"role": role, "text": text}
                if state:
                    event_data["state"] = state
                self._broadcast_sse("text_updated", event_data)
        elif msg_type == "llm":
            emotion = message.get("emotion")
            if emotion:
                self._current_emotion = emotion
                self._broadcast_sse("emotion_changed", {"emotion": emotion})
