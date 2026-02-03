import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

import yaml
from telethon import TelegramClient, events

URL_REGEX = re.compile(r"https?://\S+", re.IGNORECASE)
ASIN_PATH_REGEX = re.compile(
    r"/(?:dp|gp/product|product)/([A-Z0-9]{10})",
    re.IGNORECASE,
)
ASIN_QUERY_REGEX = re.compile(r"[?&]asin=([A-Z0-9]{10})", re.IGNORECASE)
PRICE_REGEX = re.compile(
    r"(?:€\s*(\d+[\d.,]*)|(\d+[\d.,]*)\s*(?:€|eur))",
    re.IGNORECASE,
)
DEFAULT_STORE_DOMAINS = [
    "amazon.",
    "amzn.to",
    "amzn.eu",
    "a.co",
    "amzlink.to",
    "aliexpress.",
    "s.click.aliexpress.com",
]


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\x1b[36m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[35m",
    }
    RESET = "\x1b[0m"

    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelno)
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        output = super().format(record)
        record.levelname = original_levelname
        return output


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Copy config.example.yaml to config.yaml."
        )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.lower()


def normalize_url(url: str) -> str:
    return url.strip().rstrip(").,]>'\"")


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    return [normalize_url(url) for url in URL_REGEX.findall(text)]


def extract_message_urls(message) -> list[str]:
    urls = []
    urls.extend(extract_urls(getattr(message, "message", "") or ""))

    for entity in getattr(message, "entities", []) or []:
        url = getattr(entity, "url", None)
        if url:
            urls.append(normalize_url(url))

    buttons = getattr(message, "buttons", None) or []
    for row in buttons:
        for button in row:
            url = getattr(button, "url", None)
            if url:
                urls.append(normalize_url(url))

    seen = set()
    unique = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def is_allowed_store_url(url: str, allowed_domains: list[str]) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in allowed_domains)


def extract_asin(url: str) -> str | None:
    match = ASIN_PATH_REGEX.search(url)
    if match:
        return match.group(1).upper()
    match = ASIN_QUERY_REGEX.search(url)
    if match:
        return match.group(1).upper()
    return None


def extract_price(text: str) -> float | None:
    if not text:
        return None
    match = PRICE_REGEX.search(text)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    if not value:
        return None
    value = value.replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


class DedupeStore:
    def __init__(self, db_path: str, ttl_hours: int) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = int(ttl_hours) * 3600
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS dedupe (key TEXT PRIMARY KEY, ts INTEGER)"
        )
        self.conn.commit()
        self.last_cleanup = 0

    def cleanup(self) -> None:
        now = int(time.time())
        if now - self.last_cleanup < 600:
            return
        cutoff = now - self.ttl_seconds
        self.conn.execute("DELETE FROM dedupe WHERE ts < ?", (cutoff,))
        self.conn.commit()
        self.last_cleanup = now

    def seen(self, keys: list[str]) -> bool:
        if not keys:
            return False
        self.cleanup()
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT key FROM dedupe WHERE key IN ({placeholders}) LIMIT 1"
        cursor = self.conn.execute(query, keys)
        if cursor.fetchone():
            return True
        now = int(time.time())
        self.conn.executemany(
            "INSERT OR IGNORE INTO dedupe (key, ts) VALUES (?, ?)",
            [(key, now) for key in keys],
        )
        self.conn.commit()
        return False


def build_keyword_patterns(
    values: list[str], match_mode: str
) -> list[tuple[str, re.Pattern | None]]:
    normalized = []
    for value in values or []:
        cleaned = value.strip()
        if cleaned:
            normalized.append(normalize_text(cleaned))
    patterns: list[tuple[str, re.Pattern | None]] = []
    for keyword in normalized:
        if match_mode == "substring":
            patterns.append((keyword, None))
        else:
            pattern = re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)")
            patterns.append((keyword, pattern))
    return patterns


def match_keywords(
    text: str, patterns: list[tuple[str, re.Pattern | None]], match_mode: str
) -> list[str]:
    matched = []
    for keyword, pattern in patterns:
        if match_mode == "substring":
            if keyword in text:
                matched.append(keyword)
        else:
            if pattern and pattern.search(text):
                matched.append(keyword)
    return matched


def resolve_chat_label(chat) -> tuple[str, str | None]:
    title = getattr(chat, "title", None) or getattr(chat, "username", None)
    username = getattr(chat, "username", None)
    return title or "unknown", username


def preview_text(text: str, max_chars: int = 160) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) > max_chars:
        return compact[:max_chars].rstrip() + "..."
    return compact


def setup_logging(logging_cfg: dict) -> None:
    log_level = logging_cfg.get("level", "INFO").upper()
    use_color = logging_cfg.get("color")
    if use_color is None:
        use_color = bool(sys.stdout.isatty() and not os.getenv("NO_COLOR"))
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColorFormatter("%(asctime)s %(levelname)s %(message)s", use_color)
    )
    logging.basicConfig(level=log_level, handlers=[handler])
    telethon_level = logging_cfg.get("telethon_level", "WARNING").upper()
    logging.getLogger("telethon").setLevel(telethon_level)


def resolve_session_name(telegram_cfg: dict) -> str:
    session_name = telegram_cfg.get("session_name", "dealsieve")
    session_path = telegram_cfg.get("session_path")
    if session_path:
        session_name = str(Path(session_path) / session_name)
    return session_name


def send_via_bot_api(token: str, chat_id: str | int, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": False,
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        description = body
        try:
            parsed = json.loads(body)
            description = parsed.get("description", body)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Bot API error {exc.code}: {description}") from exc


def clear_session_files(telegram_cfg: dict) -> tuple[list[str], list[str], list[str]]:
    session_name = resolve_session_name(telegram_cfg)
    candidates = []
    if session_name.endswith(".session"):
        base = session_name[: -len(".session")]
        candidates.append(session_name)
        candidates.append(base + ".session-journal")
    else:
        candidates.append(session_name + ".session")
        candidates.append(session_name + ".session-journal")

    removed = []
    locked = []
    failed = []
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                path.unlink()
                removed.append(str(path))
            except PermissionError:
                locked.append(str(path))
            except OSError as exc:
                failed.append(f"{path} ({exc})")
    return removed, locked, failed


async def run_bot(config: dict) -> None:
    telegram_cfg = config.get("telegram", {})
    filters_cfg = config.get("filters", {})
    dedupe_cfg = config.get("dedupe", {})
    output_cfg = config.get("output", {})
    logging_cfg = config.get("logging", {})

    api_id = telegram_cfg.get("api_id")
    api_hash = telegram_cfg.get("api_hash")
    session_name = telegram_cfg.get("session_name", "dealsieve")
    if not api_id or not api_hash:
        raise ValueError("api_id and api_hash are required in config.yaml")

    groups = filters_cfg.get("groups_allowlist", [])
    keyword_match = (filters_cfg.get("keyword_match") or "whole").lower()
    if keyword_match not in {"whole", "substring"}:
        keyword_match = "whole"
    include_keywords = build_keyword_patterns(
        filters_cfg.get("keywords_include", []), keyword_match
    )
    exclude_keywords = build_keyword_patterns(
        filters_cfg.get("keywords_exclude", []), keyword_match
    )
    require_store_link = filters_cfg.get("require_store_link")
    if require_store_link is None:
        require_store_link = filters_cfg.get("require_amazon_link", True)
    allowed_domains = filters_cfg.get("allowed_link_domains")
    if not allowed_domains:
        allowed_domains = DEFAULT_STORE_DOMAINS
    allowed_domains = [domain.strip().lower() for domain in allowed_domains if domain]
    price_min = filters_cfg.get("price_min")
    price_max = filters_cfg.get("price_max")

    dedupe_db = dedupe_cfg.get("db_path", "dealsieve.db")
    dedupe_ttl = dedupe_cfg.get("ttl_hours", 24)
    dedupe_store = DedupeStore(dedupe_db, dedupe_ttl)

    send_to = output_cfg.get("send_to", "me")
    include_original = output_cfg.get("include_original", True)
    max_text_chars = int(output_cfg.get("max_text_chars", 600))
    output_mode = (output_cfg.get("mode") or "telethon").lower()
    bot_token = output_cfg.get("bot_token")
    bot_chat_id = output_cfg.get("bot_chat_id")
    use_bot_api = output_mode == "bot_api" or (bot_token and bot_chat_id)
    if output_mode == "bot_api" and (not bot_token or not bot_chat_id):
        raise ValueError("output.mode=bot_api requires bot_token and bot_chat_id")

    setup_logging(logging_cfg)

    logging.info("DealSieve config loaded")
    logging.info("Monitoring %s groups", len(groups))
    logging.info("Require store link: %s", require_store_link)
    logging.info("Dedup TTL hours: %s", dedupe_ttl)
    logging.info("Output target: %s", send_to)
    logging.info("Output mode: %s", "bot_api" if use_bot_api else "telethon")
    logging.debug("Groups allowlist: %s", groups)
    logging.debug("Keyword match mode: %s", keyword_match)
    logging.debug("Include keywords: %s", [kw for kw, _ in include_keywords])
    logging.debug("Exclude keywords: %s", [kw for kw, _ in exclude_keywords])
    logging.debug("Price min: %s", price_min)
    logging.debug("Price max: %s", price_max)
    logging.debug("Allowed store domains: %s", allowed_domains)

    session_name = resolve_session_name(telegram_cfg)
    client = TelegramClient(session_name, api_id, api_hash)
    client_any = cast(Any, client)

    @client.on(events.NewMessage(chats=groups))
    async def handle_message(event: events.NewMessage.Event) -> None:
        message = event.message
        text = message.message or ""
        if not text.strip():
            logging.debug("Skip: empty message")
            return

        chat = await event.get_chat()
        chat_label, chat_username = resolve_chat_label(chat)
        logging.debug(
            "Message %s from chat %s (%s)", message.id, chat_label, event.chat_id
        )
        logging.debug("Message preview: %s", preview_text(text))

        urls = extract_message_urls(message)
        logging.debug("Extracted URLs: %s", urls)
        store_urls = [url for url in urls if is_allowed_store_url(url, allowed_domains)]
        if require_store_link and not store_urls:
            logging.debug("Skip: no allowed store link found")
            return

        normalized_text = normalize_text(text)

        matched_keywords = []
        if include_keywords:
            matched_keywords = match_keywords(
                normalized_text, include_keywords, keyword_match
            )
            if not matched_keywords:
                logging.debug("Skip: no include keywords matched")
                return
        if exclude_keywords:
            if match_keywords(normalized_text, exclude_keywords, keyword_match):
                logging.debug("Skip: excluded keyword matched")
                return

        price = extract_price(text)
        if (price_min is not None or price_max is not None) and price is None:
            logging.debug("Skip: price required but not found")
            return
        if price_min is not None and price is not None and price < float(price_min):
            logging.debug("Skip: price below minimum")
            return
        if price_max is not None and price is not None and price > float(price_max):
            logging.debug("Skip: price above maximum")
            return

        dedupe_keys = []
        for url in store_urls:
            asin = extract_asin(url)
            if asin:
                dedupe_keys.append(f"asin:{asin}")
            else:
                dedupe_keys.append(f"url:{normalize_url(url).lower()}")
        if dedupe_store.seen(dedupe_keys):
            logging.info("Duplicate offer skipped")
            logging.debug("Dedupe keys: %s", dedupe_keys)
            return

        logging.info("Match found. Sending notification.")
        logging.debug("Matched keywords: %s", matched_keywords)
        logging.debug("Store URLs: %s", store_urls)
        message_link = None
        if chat_username:
            message_link = f"https://t.me/{chat_username}/{message.id}"

        lines = ["Nuova offerta filtrata"]
        lines.append(f"Gruppo: {chat_label}")
        if message_link:
            lines.append(f"Messaggio: {message_link}")
        if store_urls:
            lines.append(f"Link: {store_urls[0]}")
        if price is not None:
            lines.append(f"Prezzo: {price:.2f} EUR")
        else:
            lines.append("Prezzo: n/d")
        if matched_keywords:
            lines.append(f"Match: {', '.join(sorted(set(matched_keywords)))}")

        if include_original:
            preview = text.strip()
            if max_text_chars and len(preview) > max_text_chars:
                preview = preview[:max_text_chars].rstrip() + "..."
            lines.append("Testo:")
            lines.append(preview)

        try:
            payload = "\n".join(lines)
            if use_bot_api:
                await asyncio.to_thread(
                    send_via_bot_api, bot_token, bot_chat_id, payload
                )
                logging.info("Notification sent via bot API to %s", bot_chat_id)
            else:
                await client.send_message(send_to, payload)
                logging.info("Notification sent to %s", send_to)
        except Exception:
            target = bot_chat_id if use_bot_api else send_to
            logging.exception("Failed to send notification to %s", target)

    try:
        await client_any.start()
    except sqlite3.OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logging.error("Session database is locked.")
            logging.error(
                "Close other bot instances or remove the .session files for this session name."
            )
            logging.error(
                "You can also change telegram.session_name to create a fresh session."
            )
            raise SystemExit(1) from exc
        raise
    logging.info("DealSieve avviato. In ascolto dei gruppi configurati.")
    await client_any.run_until_disconnected()


def main() -> None:
    parser = argparse.ArgumentParser(description="DealSieve Telegram userbot")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--clear-session",
        action="store_true",
        help="Delete the Telegram .session files and exit",
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    if args.clear_session:
        setup_logging(config.get("logging", {}))
        removed, locked, failed = clear_session_files(config.get("telegram", {}))
        if removed:
            logging.info("Removed session files:")
            for path in removed:
                logging.info("- %s", path)
        if locked:
            logging.error("Session files are locked (close other instances):")
            for path in locked:
                logging.error("- %s", path)
        if failed:
            logging.error("Failed to remove session files:")
            for item in failed:
                logging.error("- %s", item)
        if not removed and not locked and not failed:
            logging.warning("No session files found to remove.")
        return
    asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
