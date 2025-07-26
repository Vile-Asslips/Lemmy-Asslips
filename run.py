#!/usr/bin/env python3
"""
Asslips9.py – Lemmy GPT bots with custom toxic‐word filtering
            & literal newline un‑escaping
"""

import argparse
import logging
import random
import re
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import yaml
from pythorhead import Lemmy

# ------------------------------------------------------------------ #
#  SortType compatibility shim                                      #
# ------------------------------------------------------------------ #
try:
    from pythorhead.enums import SortType
except ImportError:
    try:
        from pythorhead.const import SortType
    except ImportError:
        class _EnumPlaceholder(str):
            @property
            def value(self) -> str:
                return str(self)
        class SortType:  # type: ignore
            New = _EnumPlaceholder("New")

# ------------------------------------------------------------------ #
#  Helpers                                                          #
# ------------------------------------------------------------------ #
CLEAN_TOKENS = [
    "<|soss|>", "<|sot|>", "<|eot|>", "<|sost|>", "<|eost|>",
    "<|sols|>", "<|eols|>", "<|sor|>", "<|eor|>", "<|sol|>",
    "<|eol|>", "<|eoss|>"
]
_ZWJ_RE = re.compile(r"[\u200d\uFE0F]")
_ZWSP   = "\u200b"  # zero‑width space

def clean(text: str) -> str:
    # 1) strip model‑control tokens
    for tok in CLEAN_TOKENS:
        text = text.replace(tok, " ")
    # 2) remove zero‑width joiner / variation selector
    text = _ZWJ_RE.sub("", text)
    # 3) un‑escape literal "\n\n" and "\n" into real newlines
    text = text.replace("\\n\\n", "\n\n").replace("\\n", "\n")
    # 4) treat zero‑width space as newline
    text = text.replace(_ZWSP, "\n")
    return text.strip()

def split_title_body(raw: str) -> tuple[str, str]:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Untitled 🤔", ""
    title = re.sub(r"[<>|].*?$", "", lines[0][:200]).strip() or "Untitled 🤔"
    body = "\n".join(lines[1:]).strip()
    return title, body

def iter_post_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("posts", [])
    elif isinstance(raw, list):
        for item in raw:
            if "post" in item and "creator" in item:
                yield item
            else:
                yield {"post": item, "creator": {"name": item.get("name","")}}

def iter_comment_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("comments", [])
    elif isinstance(raw, list):
        for item in raw:
            if "comment" in item and "creator" in item:
                yield item
            else:
                yield {"comment": item, "creator": {"name": item.get("creator_name","")}}

# ------------------------------------------------------------------ #
#  Bot Thread                                                       #
# ------------------------------------------------------------------ #
class BotThread(threading.Thread):
    def __init__(self, bot_cfg: MappingProxyType, global_cfg: MappingProxyType):
        super().__init__(daemon=True, name=bot_cfg["name"])
        self.cfg = bot_cfg
        self.global_cfg = global_cfg
        self.log = logging.getLogger(bot_cfg["name"])

        # Lemmy login
        self.lemmy = Lemmy(global_cfg["instance"])
        self.lemmy.log_in(bot_cfg["username"], bot_cfg["password"])
        self.community_id = self.lemmy.discover_community(global_cfg["community"])

        # Model + tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(bot_cfg["model"])
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(bot_cfg["model"])
        self.model.eval()

        # Custom toxic‑word filtering (from config.yaml)
        # config.yaml must contain a list under key `toxic_words`
        self.toxic_words = [w.casefold() for w in global_cfg.get("toxic_words", [])]

        # Timing & config
        self.freq_s = float(bot_cfg.get("postfreq", 1)) * 3600
        self.initial_post = bool(global_cfg.get("initial_post", False))
        self.roll_needed = int(global_cfg.get("comment_roll", 70))
        self.max_replies = int(global_cfg.get("max_replies", 5))
        self.delay_min = float(global_cfg.get("reply_delay_min", 5))
        self.delay_max = float(global_cfg.get("reply_delay_max", 12))
        self.last_post_at = 0.0

        self.stop_event = threading.Event()

    def _is_toxic(self, txt: str) -> bool:
        """Return True if any toxic word appears in txt (case‑insensitive)."""
        low = txt.casefold()
        return any(w in low for w in self.toxic_words)

    def _gen(self, prompt: str) -> str:
        import warnings
        inputs = self.tokenizer(prompt, return_tensors="pt")
        ids = inputs.input_ids
        attn = inputs.get("attention_mask", None)

        # empty‑prompt fallback
        if not prompt.strip() or ids.size(1) == 0:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            ids = torch.tensor([[bos]], device=ids.device)
            prompt_len = 1
            attn = torch.ones_like(ids)
        else:
            prompt_len = ids.size(1)

        max_new = max(16, 384 - prompt_len)
        for _ in range(4):
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    out = self.model.generate(
                        ids,
                        attention_mask=attn,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            gen_ids = out[0, prompt_len:]
            txt = clean(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
            if txt and not self._is_toxic(txt):
                return txt
        return ""

    def _post(self, title: str, body: str) -> int | None:
        try:
            res = self.lemmy.post.create(self.community_id, title, body=body)
            pid = res["post_view"]["post"]["id"]
            self.log.info("Posted: %s", title)
            return pid
        except Exception:
            self.log.exception("post failed")
            return None

    def _comment(self, post_id: int, content: str, parent_id: int | None = None) -> None:
        try:
            self.lemmy.comment.create(post_id, content, parent_id=parent_id)
            self.log.info("Commented on %d", post_id)
        except Exception:
            self.log.exception("comment failed")

    def _attempt_replies(self, sources: list[dict[str, Any]]) -> None:
        attempts = 0
        for src in sources:
            if attempts >= self.max_replies:
                break
            # dice‑roll gate
            if random.randint(1, 100) < self.roll_needed:
                continue

            # --- build a reply prompt that won't just continue the last sentence ---
            prompt = src["text"].strip() + "\nReply:\n"

            # --- generate up to 3 tries, skipping empties or echoes ---
            reply = ""
            for _ in range(3):
                raw = self._gen(prompt).strip()
                # if model literally echoes "Reply:", strip it off
                if raw.lower().startswith("reply:"):
                    raw = raw[len("reply:"):].strip()
                # ensure it didn't just echo the source text
                if raw and raw.lower() != src["text"].lower():
                    reply = raw
                    break

            if not reply:
                continue

            # --- now post the clean reply ---
            self._comment(src["post_id"], reply, parent_id=src["parent_id"])
            attempts += 1
            time.sleep(random.uniform(self.delay_min, self.delay_max))

    def run(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()

            # (1) Post new thread – immediate on startup if initial_post=True
            if self.initial_post or (now - self.last_post_at) >= self.freq_s:
                # Try to generate a title up to 3 times
                title = ""
                for _ in range(3):
                    raw = self._gen("")
                    c = raw.splitlines()[0][:200].strip()
                    c = re.sub(r"[<>|].*?$", "", c)
                    if c and c != "Untitled 🤔":
                        title = c
                        break
                # If still empty, seed the model with "Vile Asslips"
                if not title:
                    seed = "Vile Asslips"
                    fb = self._gen(seed).splitlines()[0][:200].strip()
                    title = fb or seed

                # generate body by using title as the prompt
                body = ""
                for _ in range(3):
                    candidate = self._gen(title).strip()
                    # ensure it didn’t just echo the title
                    if candidate and candidate.lower() != title.lower():
                        body = candidate
                        break
                if not body:
                    body = " "  # Lemmy requires non‑empty

                self._post(title, body)
                self.last_post_at = now
                self.initial_post = False  # only run once

            # 2) Scan & reply every loop
            feed = self.lemmy.post.list(
                page=1, limit=self.max_replies * 3,
                sort=SortType.New, community_id=self.community_id
            )
            posts = [
                {"post_id": pv["post"]["id"],
                 "text": pv["post"]["name"]+"\n"+pv["post"].get("body",""),
                 "parent_id": None}
                for pv in iter_post_views(feed)
                if pv["creator"]["name"] != self.cfg["username"]
            ]
            self._attempt_replies(posts)

            cfeed = self.lemmy.comment.list(
                community_id=self.community_id,
                sort=SortType.New, page=1, limit=self.max_replies * 3
            )
            comments = [
                {"post_id": cv["comment"]["post_id"],
                 "text": cv["comment"]["content"],
                 "parent_id": cv["comment"]["id"]}
                for cv in iter_comment_views(cfeed)
                if cv["creator"]["name"] != self.cfg["username"]
            ]
            self._attempt_replies(comments)

            time.sleep(5)


# ------------------------------------------------------------------ #
#  Entrypoint                                                      #
# ------------------------------------------------------------------ #
def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    log_dir = Path(cfg.get("log_dir","logs"))
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if cfg.get("debug") else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(
            log_dir / f"asslips6_{int(time.time())}.log", encoding="utf-8"
        )]
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logging.getLogger().addHandler(console)

    threads = [
        BotThread(MappingProxyType(b), MappingProxyType(cfg))
        for b in cfg["bots"]
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[CTRL‑C] Shutting down…")
        for t in threads:
            t.stop_event.set()
        for t in threads:
            t.join()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="Path to YAML config")
    main(ap.parse_args().config)
