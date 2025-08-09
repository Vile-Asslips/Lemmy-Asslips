#!/usr/bin/env python3
"""
Asslips17c.py – Lemmy GPT bots with:
  • adaptive token budgeting
  • externalized, diversified fallback templates (expanded)
  • direct grounding: title/body or comment in prompt
  • dynamic few-shot + keyword grounding
  • staged temperature/top_p fallback with best-candidate retention
  • refinement and focused pass before external fallback
  • aggressive completion and final finish-guard
  • prompt truncation to prevent overflow
  • suppression of control tokens & cleaning (including literal escape removal and control sequence stripping)
  • thread-history memory via BotDB
  • per-thread fallback deduplication
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
from collections import deque
from pythorhead import Lemmy
from bot_db.db import BotDB

# SortType compatibility shim
try:
    from pythorhead.enums import SortType
except ImportError:
    try:
        from pythorhead.const import SortType
    except ImportError:
        class _EnumPlaceholder(str):
            @property
            def value(self): return str(self)
        class SortType:
            New = _EnumPlaceholder("New")


# -------------------- fallback manager -------------------- #
class FallbackManager:
    def __init__(self, path: str, history_size: int = 5):
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        self.templates: list[str] = data.get("templates", []) or []
        if not self.templates:
            raise ValueError(f"No fallback templates found in {path}")
        self.history_size = history_size
        self.recent: dict[Any, deque] = {}  # post_id -> deque of recent template indices

    def _related(self, topics: list[str]) -> str:
        if len(topics) >= 2:
            return topics[1]
        elif topics:
            return topics[0]
        return ""

    def choose(self, topics: list[str], post_id: Any) -> str:
        topic = topics[0] if topics else ""
        related = self._related(topics) if topics else ""
        all_indices = list(range(len(self.templates)))
        used = self.recent.get(post_id, deque(maxlen=self.history_size))
        candidates = [i for i in all_indices if i not in used]
        if not candidates:
            candidates = all_indices
        idx = random.choice(candidates)
        template = self.templates[idx]
        # natural-language join: "a, b and c"
        if len(topics) > 1:
            topics_joined = ", ".join(topics[:-1]) + " and " + topics[-1]
        else:
            topics_joined = topics[0] if topics else ""
        filled = (template
                  .replace("{topic}", topic or "")
                  .replace("{related}", related or "")
                  .replace("{topics}", topics_joined))

        if post_id not in self.recent:
            self.recent[post_id] = deque(maxlen=self.history_size)
        self.recent[post_id].append(idx)
        return filled


# -------------------- helpers -------------------- #
CLEAN_TOKENS = [
    "<|soss|>", "<|sot|>", "<|eot|>", "<|sost|>", "<|eost|>",
    "<|sols|>", "<|eols|>", "<|sor|>", "<|eor|>", "<|sol|>",
    "<|eol|>", "<|eoss|>", "<|pop|>", "<|pop-up|>", "<|pop!|>"
]
_ZWJ_RE = re.compile(r"[\u200d\uFE0F]")
_ZWSP = "\u200b"  # zero-width space

def clean(text: str) -> str:
    # strip known control tokens
    for tok in CLEAN_TOKENS:
        text = text.replace(tok, " ")
    # remove any <|...|> style control sequences (non-greedy)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    # hard-kill any stray <|pop ... |> style tokens
    text = re.sub(r"(?is)<\|\s*pop[^|>]*\|>", "", text)
    # strip standalone 'pop/pops/popup/pop-up' but keep 'population'/'popular'
    text = re.sub(r"(?i)\bpop(?!ulation|ular)(?:-?up|s)?\b", "", text)
    # remove zero-width joiner / variation selectors
    # keep real ZWJ so emoji sequences stay intact
    # text = _ZWJ_RE.sub("", text)
    # strip literal escape sequences like \u200d, \U0001F600, \xA0, and even bare 'u200d'
    text = re.sub(r"(?i)\\[uU][0-9a-f]{2,8}", "", text)  # \uXXXX or \UXXXXXXXX
    text = re.sub(r"(?i)\\x[0-9a-f]{2}", "", text)       # \xHH
    text = re.sub(r"\\[0-7]{3}", "", text)               # octal \123
    text = re.sub(r"(?i)\\?u[0-9a-f]{2,6}", "", text)    # 'u200d' or '\u200d'

    # remove literal escape sequences and normalize newlines
    text = text.replace("\\u200b", "").replace("\\u200d", "").replace("\\200d", "")
    text = text.replace("\\n\\n", "\n\n").replace("\\n", "\n")
    text = text.replace(_ZWSP, "")
    # remove stray angle-bracket tokens like <esol>, <eso>, <sor>, etc.
    text = re.sub(r"<(?:esol|eso|eor|sol|sor)>", "", text, flags=re.IGNORECASE)
    # collapse whitespace
    return " ".join(text.strip().split())


def split_title_body(raw: str) -> tuple[str, str]:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return "Untitled 🤔", ""
    title = re.sub(r"[<>|].*?$", "", lines[0][:200]).strip() or "Untitled 🤔"
    body = "\n".join(lines[1:]).strip()
    return title, body


def extract_keywords(text: str, max_words: int = 5) -> list[str]:
    stopset = {
        "the", "and", "a", "an", "of", "to", "in", "for", "on", "with",
        "that", "this", "it", "is", "was", "you", "have", "has", "but",
        "not", "what", "kind", "more", "than", "also", "they", "just",
        "i", "me", "my", "our", "your"
    }
    words = re.findall(r"\b\w+'\w+|\b\w+\b", text.lower())
    filtered = [w for w in words if w not in stopset and len(w) >= 3]
    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    sorted_keywords = sorted(freq.keys(), key=lambda k: (-freq[k], len(k)))
    return sorted_keywords[:max_words]


def make_few_shot_example(topics: list[str]) -> str:
    primary = topics[0] if topics else "the topic"
    good = f"I hear you about {primary}. Here's a suggestion related to {primary}."
    bad = "I love pizza and cats."
    return f"Good example reply: \"{good}\"\nBad example reply: \"{bad}\"\n\n"


def is_relevant_enough(reply: str, topics: list[str]) -> bool:
    low = reply.lower()
    return any(re.search(rf"\b{re.escape(t.lower())}\b", low) for t in topics)


def iter_post_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("posts", [])
    else:
        for item in raw or []:
            if "post" in item and "creator" in item:
                yield item
            else:
                yield {"post": item, "creator": {"name": item.get("name", "")}}


def iter_comment_views(raw: Any):
    if isinstance(raw, dict):
        yield from raw.get("comments", [])
    else:
        for item in raw or []:
            if "comment" in item and "creator" in item:
                yield item
            else:
                yield {"comment": item, "creator": {"name": item.get("creator_name", "")}}


# -------------------- BotThread -------------------- #
class BotThread(threading.Thread):
    def __init__(self, bot_cfg: MappingProxyType, global_cfg: MappingProxyType, fallback_manager: FallbackManager):
        super().__init__(daemon=True, name=bot_cfg.get("name", "bot"))
        self.cfg = bot_cfg
        self.global_cfg = global_cfg
        self.fallback_manager = fallback_manager
        self.log = logging.getLogger(bot_cfg.get("name", "bot"))

        self.lemmy = Lemmy(global_cfg.get("instance"))
        self.lemmy.log_in(bot_cfg.get("username"), bot_cfg.get("password"))
        self.community_id = self.lemmy.discover_community(global_cfg.get("community"))

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(bot_cfg.get("model"))
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(bot_cfg.get("model"))
        self.model.eval()

        bad_words_ids = []
        for tok in CLEAN_TOKENS:
            toks = self.tokenizer.encode(tok, add_special_tokens=False)
            if toks:
                bad_words_ids.append(toks)
        self.bad_words_ids = bad_words_ids

        # Optional: forbid unicode-escape starters so the model can't emit them
        if self.cfg.get("ban_unicode_escapes", True):
            for w in ["\\u", "\\U", "\\x"]:
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    self.bad_words_ids.append(ids)

        # Optional hard-ban of 'pop' spam during generation (toggle per-bot in config)
        if self.cfg.get("ban_pop", True):
            for w in ["pop", "pops", "popup", "pop-up", "Pop", "POP", "Pop!", "pop!"]:
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    self.bad_words_ids.append(ids)

        self.toxic_words = [w.casefold() for w in global_cfg.get("toxic_words", [])]

        self.freq_s = float(bot_cfg.get("postfreq", 1)) * 3600
        self.initial_post = bool(global_cfg.get("initial_post", False))
        self.roll_needed = int(global_cfg.get("comment_roll", 70))
        self.max_replies = int(bot_cfg.get("max_replies", 5))
        self.delay_min = float(bot_cfg.get("reply_delay_min", 5))
        self.delay_max = float(bot_cfg.get("reply_delay_max", 12))
        self.base_max_new_tokens = int(bot_cfg.get("max_new_tokens", 64))
        self.current_max_new_tokens = self.base_max_new_tokens
        self.min_new_tokens_allowed = int(bot_cfg.get("min_new_tokens", 16))
        self.max_new_tokens_cap = int(bot_cfg.get("max_new_tokens_cap", self.base_max_new_tokens))

        self.last_post_at = 0.0
        self.db = BotDB(bot_cfg.get("username"), self.community_id)
        self.stop_event = threading.Event()

    def _is_toxic(self, txt: str) -> bool:
        return any(w in txt.casefold() for w in self.toxic_words)

    def _is_incomplete_reply(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        stripped = text.strip()
        if stripped.endswith(('.', '!', '?', '...')):
            return False
        last_word = stripped.split()[-1].lower()
        if last_word in {'and', 'or', 'but', 'with', 'to', 'for', 'because', 'so', 'if', 'that', 'which', 'when', 'while', 'as', 'than', 'then'}:
            return True
        if len(stripped.split()) < 8:
            return True
        return False

    def adjust_token_budget(self, last_reply: str, was_empty: bool):
        if was_empty:
            self.current_max_new_tokens = max(self.min_new_tokens_allowed, int(self.current_max_new_tokens * 0.75))
        else:
            incomplete = self._is_incomplete_reply(last_reply)
            if incomplete:
                self.current_max_new_tokens = min(self.max_new_tokens_cap, int(self.current_max_new_tokens * 1.25))
            else:
                if self.current_max_new_tokens > self.base_max_new_tokens:
                    self.current_max_new_tokens = max(self.base_max_new_tokens, int(self.current_max_new_tokens * 0.9))
                else:
                    self.current_max_new_tokens = max(self.min_new_tokens_allowed, int(self.current_max_new_tokens * 0.95))

    def _gen(self, prompt: str, skip_toxic: bool = False, temperature: float | None = None,
             top_p: float | None = None, override_new_tokens: int | None = None) -> str:
        if hasattr(self.model.config, 'n_ctx'):
            context_window = self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            context_window = self.model.config.max_position_embeddings
        else:
            context_window = None

        if context_window is not None:
            if override_new_tokens is not None:
                max_prompt_len = max(1, context_window - override_new_tokens - 1)
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_prompt_len)
            else:
                max_prompt_len = max(1, context_window - 1)
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_prompt_len)
        else:
            inputs = self.tokenizer(prompt, return_tensors='pt')
        ids = inputs.input_ids
        attn = inputs.get('attention_mask', None)

        if not prompt.strip() or ids.size(1) == 0:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            ids = torch.tensor([[bos]], device=ids.device)
            prompt_len = 1
            attn = torch.ones_like(ids)
        else:
            prompt_len = ids.size(1)

        if context_window is not None:
            if override_new_tokens is not None:
                allowed_new = min(override_new_tokens, context_window - prompt_len - 1)
            else:
                allowed_new = min(self.current_max_new_tokens, context_window - prompt_len - 1)
            allowed_new = max(1, allowed_new)
        else:
            allowed_new = override_new_tokens if override_new_tokens is not None else self.current_max_new_tokens

        min_new = 20 if allowed_new >= 20 else allowed_new
        temp = temperature if temperature is not None else 0.65
        tp = top_p if top_p is not None else 0.75

        for _ in range(4):
            with torch.no_grad():
                out = self.model.generate(
                    ids,
                    attention_mask=attn,
                    max_new_tokens=allowed_new,
                    min_new_tokens=min_new,
                    do_sample=True,
                    temperature=temp,
                    top_p=tp,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=self.bad_words_ids,
                )
            gen_ids = out[0, prompt_len:]
            txt = clean(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
            if txt and (skip_toxic or not self._is_toxic(txt)):
                return txt
        return ""

    def _try_complete(self, candidate: str, topic_str: str, temp: float, tp: float) -> str:
        """
        If candidate looks incomplete, attempt up to two refinements to finish the thought,
        giving more token budget.
        """
        for _ in range(2):
            if not self._is_incomplete_reply(candidate):
                break
            continuation_prompt = (
                f"The previous reply was cut off. Here is what it gave:\n\"{candidate}\"\n"
                f"Finish the thought in one or two sentences, referencing at least one of these topics: {topic_str}. "
                "Do not repeat the whole previous reply; just continue and conclude the idea. Reply:"
            )
            more = self._gen(continuation_prompt, skip_toxic=False, temperature=temp, top_p=tp, override_new_tokens=128).strip()
            if more.lower().startswith("reply:"):
                more = more[len("reply:"):].strip()
            if not more:
                break
            candidate = candidate.rstrip(" .,!?:;") + " " + more.lstrip()
        return candidate

    def _post(self, title: str, body: str) -> int | None:
        # belt-and-suspenders: ensure no links slip into the title here either
        title = re.sub(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', '', title)
        title = re.sub(r'!{9,}',  '!!!!!!!!', title)   # cap ! at 8
        title = re.sub(r'\?{9,}', '????????', title)   # cap ? at 8
        title = title.strip()

        try:
            res = self.lemmy.post.create(self.community_id, title, body=body, nsfw=True)
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
            try:
                if attempts >= self.max_replies:
                    break
                if random.randint(1, 100) < self.roll_needed:
                    continue

                post_id    = src.get("post_id")
                is_comment = src.get("parent_id") is not None
                raw_text   = src.get("text", "").strip()

                history = self.db.get_thread_history(post_id, limit=3)
                history_str = ""
                if history:
                    reversed_hist = list(reversed(history))
                    history_lines = []
                    for author, gen in reversed_hist:
                        snippet = gen.replace("\n", " ").strip()
                        history_lines.append(f"{author}: {snippet}")
                    history_str = "Recent thread context:\n" + "\n".join(history_lines) + "\n\n"

                # build topics (and forced words) differently for posts vs. comments
                orig_topics = extract_keywords(raw_text, max_words=5)
                if not is_comment:
                    # for posts: inject 1–2 words from title & body
                    title, body = split_title_body(raw_text)
                    title_words = re.findall(r"\b[A-Za-z]{4,}\b", title)[:2]
                    body_words  = re.findall(r"\b[A-Za-z]{4,}\b", body)[:2]
                    topics      = orig_topics + title_words + body_words
                else:
                    # for comments: no forced words, but still define the vars
                    title_words = []
                    body_words  = []
                    topics      = orig_topics
                topic_str = ", ".join(topics) if topics else "general"
                few_shot  = make_few_shot_example(topics)

                # build prompt
                if is_comment:
                    prompt = (
                        f"{few_shot}"
                        f"You are replying to a comment in the Asslips community. Original comment:\n\"{raw_text}\"\n\n"
                        f"Main topics: {topic_str}\n"
                        "Instructions:\n"
                        "1. Briefly acknowledge something specific from that comment (reference at least one of the topics above) in one sentence.\n"
                        "2. Then provide a relevant, on-topic response or continuation.\n"
                        "3. Do not repeat more than 10 words verbatim; paraphrase in your own words.\n"
                        "4. Keep it to 2-3 sentences and finish the thought. Reply:"
                    )
                else:
                    title, body = split_title_body(raw_text)
                    if len(body) > 1000:
                        body = body[:1000].rsplit(" ", 1)[0] + "..."
                    prompt = (
                        f"{few_shot}"
                        f"You are participating in the Asslips community. Below is the original post:\n\n"
                        f"Title: \"{title}\"\n"
                        f"Body: \"{body}\"\n\n"
                        f"Main topics: {topic_str}\n"
                        "Instructions:\n"
                        "1. Briefly acknowledge something specific from the title or body (reference at least one of the topics above) in one sentence.\n"
                        "2. Then give an on-topic follow-up, advice, or commentary related to that.\n"
                        "3. Do not repeat more than 10 words verbatim from the original; use your own phrasing.\n"
                        "4. Keep it coherent, avoid unrelated rambling, and end with a complete thought. Limit to about 2-3 sentences. Reply:"
                    )

                reply = ""
                best_candidate = ""
                best_score = -1

                # staged attempts with dynamic strictness (per-bot settings)
                strict = self.cfg.get("strict_mode", False)
                mention_tries   = int(self.cfg.get("mention_strict_tries",   2))
                complete_tries  = int(self.cfg.get("complete_strict_tries",  2))
                len_init        = int(self.cfg.get("length_initial",       10))
                len_min         = int(self.cfg.get("length_min",            0))
                max_tries       = 5 if is_comment else 3

                for attempt in range(max_tries):
                    # pick temperature/top_p per stage
                    if attempt == 0:
                        temp, tp = 0.65, 0.75
                    elif attempt == 1:
                        temp, tp = 0.70, 0.80
                    else:
                        temp, tp = 0.80, 0.90

                    # generate
                    candidate = self._gen(prompt, skip_toxic=False, temperature=temp, top_p=tp).strip()
                    if candidate.lower().startswith("reply:"):
                        candidate = candidate[len("reply:"):].strip()
                    if not candidate or candidate.lower() == raw_text.lower():
                        continue

                    # finish cut-off replies
                    candidate = self._try_complete(candidate, topic_str, temp, tp)

                    # comment: one instruction-guided retry if off-topic/empty
                    if is_comment and (not candidate or not is_relevant_enough(candidate, topics)):
                        inst_prompt = (
                            f"You are replying to a comment in the Asslips community.\n"
                            f"Original comment: \"{raw_text}\"\n\n"
                            "Instructions:\n"
                            "1. Briefly acknowledge something specific from that comment.\n"
                            "2. Provide a relevant, on-topic response or continuation.\n"
                            "3. Do not repeat more than 10 words verbatim; paraphrase.\n"
                            "4. Keep it coherent and finish with a complete thought in 2-3 sentences.\n"
                            "Reply:"
                        )
                        inst_candidate = self._gen(inst_prompt, skip_toxic=False, temperature=temp, top_p=tp).strip()
                        if inst_candidate:
                            candidate = inst_candidate

                    # decide which rules to enforce this round
                    enforce_mention  = strict and (attempt < mention_tries)  and not is_comment
                    enforce_complete = strict and (attempt < complete_tries)
                    length_threshold = max(
                        len_min,
                        len_init - int((len_init - len_min) * (attempt / max(1, max_tries - 1)))
                    )

                    # 1) forced-mention on posts
                    if enforce_mention:
                        if not any(re.search(rf"\b{re.escape(w)}\b", candidate, re.IGNORECASE) for w in title_words + body_words):
                            continue

                    # 2) complete-thought check
                    if enforce_complete and self._is_incomplete_reply(candidate):
                        continue

                    # 3) length requirement
                    if len(candidate.split()) < length_threshold:
                        continue

                    # simple scoring
                    score = 0
                    if is_relevant_enough(candidate, topics):
                        score += 2
                    if not self._is_incomplete_reply(candidate):
                        score += 1
                    if len(candidate.split()) >= length_threshold:
                        score += 1

                    # remember best
                    if score > best_score:
                        best_score     = score
                        best_candidate = candidate

                    # early accept if fully past enforced rules
                    if not enforce_mention and not (enforce_complete and self._is_incomplete_reply(candidate)) \
                       and len(candidate.split()) >= length_threshold:
                        reply = candidate
                        break

                # refinement pass
                if not reply and best_candidate:
                    refine_prompt = (
                        f"Improve the following reply to make it more on-topic, include at least one of the main topics: {topic_str}, "
                        f"and finish the thought. Previous reply: \"{best_candidate}\". Reply:"
                    )
                    refined = self._gen(refine_prompt, skip_toxic=False, temperature=0.7, top_p=0.85, override_new_tokens=128).strip()
                    if refined.lower().startswith("reply:"):
                        refined = refined[len("reply:"):].strip()
                    if refined and is_relevant_enough(refined, topics) and not self._is_incomplete_reply(refined):
                        reply = refined
                    else:
                        reply = best_candidate

                # focused low-entropy pass
                if not reply:
                    focused_prompt = f"{prompt}\n\nNow respond in one concise sentence that references at least one of the main topics and completes the thought."
                    focused = self._gen(focused_prompt, skip_toxic=False, temperature=0.3, top_p=0.9).strip()
                    if focused.lower().startswith("reply:"):
                        focused = focused[len("reply:"):].strip()
                    if focused and is_relevant_enough(focused, topics):
                        reply = focused

                # final finish guard if still incomplete
                if reply and self._is_incomplete_reply(reply):
                    finish_prompt = f"That last reply was cut off. Finish it cleanly: \"{reply}\". Reply:"
                    extra = self._gen(finish_prompt, skip_toxic=False, temperature=0.5, top_p=0.9, override_new_tokens=64).strip()
                    if extra.lower().startswith("reply:"):
                        extra = extra[len("reply:"):].strip()
                    if extra:
                        reply = reply.rstrip(" .,!?:;") + " " + extra.lstrip()

                # last-resort fallback
                if not reply:
                    bot_fp = self.cfg.get("fallback_prompt")
                    if bot_fp:
                        # gather candidate keywords (preserve order, dedupe)
                        if not is_comment:
                            pool = [w for w in (title_words + body_words) if w]
                        else:
                            pool = [w for w in (orig_topics) if w]
                        if not pool:
                            pool = [w for w in (topics) if w]
                        pool = list(dict.fromkeys(pool))  # dedupe, keep order

                        # build natural-language list safely
                        if not pool:
                            dynamic_topics = "this"
                        elif len(pool) == 1:
                            dynamic_topics = pool[0]
                        else:
                            dynamic_topics = ", ".join(pool[:-1]) + " and " + pool[-1]

                        # fill and generate
                        prompt = bot_fp.format(topics=dynamic_topics)
                        gen = self._gen(prompt, skip_toxic=False, temperature=0.75, top_p=0.9).strip()
                        reply = gen or prompt
                        self.log.debug("Dynamic multi-topic fallback for %s → %r", post_id, reply)
                    else:
                        if is_comment:
                            if best_candidate:
                                reply = best_candidate
                                self.log.debug("Best candidate used for comment %s → %r", post_id, reply)
                            else:
                                self.log.debug("No candidate for comment %s → skipping", post_id)
                                continue
                        else:
                            reply = self.fallback_manager.choose(topics, post_id)
                            self.log.debug("YAML fallback used for post %s → %s", post_id, reply)

                # dedupe: don't repost same
                last_hist = self.db.get_thread_history(post_id, limit=1)
                if last_hist:
                    last_author, last_gen = last_hist[-1]
                    if last_author == self.cfg.get("username") and last_gen.strip() == reply.strip():
                        continue

                # clean up + post
                reply = reply.lstrip(' \n\r\t-–—:;,.!?\"\'')
                if reply:
                    reply = reply[0].upper() + reply[1:]
                reply = clean(reply)
                self._comment(post_id, reply, parent_id=src.get("parent_id"))
                attempts += 1
                time.sleep(random.uniform(self.delay_min, self.delay_max))

            except Exception:
                self.log.exception("Error handling item: %r", src)
                continue


    def run(self) -> None:
        while not self.stop_event.is_set():
            now = time.time()

            if self.initial_post or (now - self.last_post_at) >= self.freq_s:
                title = ""
                fallback_prompts = [
                    "Vile Asslips", "AgentGiga", "Krobix897",
                    "Buttholes", "Turds", "farts", "farting", "fart",
                    "dingle berry", "Lemmy", "Sour Asslips",
                ]
                for p in [""] + fallback_prompts:
                    cand = self._gen(p).splitlines()[0][:200].strip() if p else ""
                    if cand:
                        cand = re.sub(r"[<>|].*?$", "", cand)
                    if cand and cand != "Untitled 🤔":
                        title = cand
                        break
                if not title:
                    title = "Community of Asslips"

                body = ""
                for _ in range(3):
                    b = self._gen(title).strip()
                    if b and b.lower() != title.lower():
                        body = b
                        break
                if not body:
                    body = " "

                # strip URLs from the title and move them into the body
                urls = re.findall(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', title)
                title = re.sub(r'(https?://\S+|www\.\S+|\[[^\]]+\]\(https?://[^\s)]+\))', '', title)

                # keep dramatic punctuation but cap at 8 if you want
                title = re.sub(r'\bsource\s*:?\s*$', '', title, flags=re.I)
                title = re.sub(r'!{9,}',  '!!!!!!!!', title)   # cap ! at 8
                title = re.sub(r'\?{9,}', '????????', title)   # cap ? at 8
                title = re.sub(r'\s{2,}', ' ', title).strip()

                # if title emptied by scrubbing, fall back to a body snippet
                if not title:
                    title = ' '.join(body.split()[:8]) or 'Quick thought'

                # if we stripped links, append them to the body as sources
                if urls:
                    plain_urls = [re.sub(r'^\[.*?\]\((https?://[^\s)]+)\)$', r'\1', u) for u in urls]
                    body = (body.rstrip() + "\n\nSources: " + " ".join(plain_urls)).strip()

                # post and update the gate timers
                self._post(title, body)
                self.last_post_at = now
                self.initial_post = False


            feed = self.lemmy.post.list(
                page=1,
                limit=self.max_replies * 3,
                sort=SortType.New,
                community_id=self.community_id,
            )
            posts = [
                {
                    "post_id": pv["post"]["id"],
                    "text": pv["post"]["name"] + "\n" + pv["post"].get("body", ""),
                    "parent_id": None,
                }
                for pv in iter_post_views(feed)
            ]
            self._attempt_replies(posts)

            cfeed = self.lemmy.comment.list(
                community_id=self.community_id,
                sort=SortType.New,
                page=1,
                limit=self.max_replies * 3,
            )
            comments = [
                {
                    "post_id": cv["comment"]["post_id"],
                    "text": cv["comment"]["content"],
                    "parent_id": cv["comment"]["id"],
                }
                for cv in iter_comment_views(cfeed)
            ]
            self._attempt_replies(comments)

            time.sleep(5)


# -------------------- entrypoint -------------------- #
def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    fallback_path = cfg.get("fallback_file", "fallbacks_expanded.yaml")
    history_size = int(cfg.get("fallback_history_size", 5))
    fm = FallbackManager(fallback_path, history_size=history_size)

    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if cfg.get("debug") else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_dir / f"asslips17c_{int(time.time())}.log", encoding="utf-8")],
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(console)

    threads = [BotThread(MappingProxyType(b), MappingProxyType(cfg), fm) for b in cfg["bots"]]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[CTRL-C] Shutting down...")
        for t in threads:
            t.stop_event.set()
        for t in threads:
            t.join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml")
    main(ap.parse_args().config)