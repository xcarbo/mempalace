"""Cross-encoder second-stage scoring.

The bi-encoder path in :mod:`mempalace.rerank` embeds the query and each
candidate independently and compares the two vectors. A cross-encoder instead
reads the query and the candidate TOGETHER and answers one question: does this
document answer this query. That is a strictly stronger signal, and it is the
one that matters for the paraphrase regime, where the query and the drawer
share almost no vocabulary and two independent embeddings have nothing to
match on.

Protocol. Qwen3-Reranker (and the family it belongs to) is not an instruct
model and must not be driven through a chat endpoint: its chat template forces
a reasoning preamble, so the verdict never arrives inside a sane token budget
(measured — the answer lands in ``reasoning_content`` and ``content`` comes back
empty). It is driven through the RAW completion endpoint with the model's own
template, capped at one token, reading the logprobs of ``yes`` against ``no``:

    score = softmax([logprob(yes), logprob(no)])[0]

Which is a real probability in [0, 1], not a cosine, and separates far more
cleanly than one. Measured against the live palace, one query over four
candidates: exact answer 0.978, same-bug-different-detail 0.163, same-project
different-topic 0.000, unrelated 0.000.

Cost. One HTTP round trip per candidate rather than one for the whole pool, so
this is seconds where the bi-encoder is milliseconds: ~7.6s for 32 real drawers
against a 4B reranker on a laptop GPU over the tailnet. Concurrency has a knee
and it moves with the server's context window, so MAX_WORKERS is a measured
default rather than "as many as there are candidates" — see its note.

That cost is why this is not a drop-in replacement for the bi-encoder in an
interactive search. It is affordable on a deep pool offline, and on a shallow
pool online.

Fail-soft by contract, exactly like the bi-encoder path: any error anywhere
returns None and the caller keeps first-stage order. Search never breaks
because a reranker is unreachable, asleep, or slow.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import logging
import math
import os
import urllib.error
import urllib.request

logger = logging.getLogger("mempalace_mcp")

URL_ENV = "MEMPALACE_RERANK_URL"
MODEL_ENV = "MEMPALACE_RERANK_MODEL"
KEY_FILE_ENV = "MEMPALACE_RERANK_KEY_FILE"

# Matches the endpoint's measured width, which depends on how the far side was
# loaded: with a 512-token context 8 was fastest and 16 was slower, but at 8192
# tokens 16 wins clearly (32 real drawers: 393 ms/doc at 8, 238 at 16, 269 at
# 24). The knee moves with the server's context and slot count, so this is a
# measured default for the current endpoint, not a universal constant.
MAX_WORKERS = 16
# Per-candidate text cap. The judgement is made on the opening of a drawer;
# sending more costs prompt tokens on every candidate and did not change the
# verdict in spot checks.
MAX_TEXT_CHARS = 2000
# Floor for the shrink-on-context-rejection retry in _score_one. Below this
# there is not enough of a drawer left to judge, so failing is more honest.
MIN_TEXT_CHARS = 400
# Attempts per candidate: the original plus halvings. 2000 -> 1000 -> 500 -> 400
# reaches the floor, which covers a 512-token window.
MAX_SHRINKS = 4
# One candidate, one token. A hung endpoint must not stall search: the caller
# already treats None as "keep first-stage order".
TIMEOUT_SECONDS = 20.0

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_INSTRUCT = "Given a search query, retrieve relevant passages"

# Casing variants count as the same verdict — the model spreads probability
# across them, and folding them in is worth ~0.02 on a confident pair.
_YES = ("yes", "Yes", "YES")
_NO = ("no", "No", "NO")
# Logprob assumed for a verdict token absent from the top-k window. Low enough
# to read as "not considered", finite so the softmax stays defined.
_ABSENT = -20.0


def _api_key() -> str | None:
    """Read the bearer token from the file named by ``MEMPALACE_RERANK_KEY_FILE``.

    A FILE, not the value: an environment variable set in a shell profile never
    reaches cron or launchd, which is the same trap the palace outbox secret
    fell into (fixed in c9520e8). Absent or unreadable means no auth header,
    which is correct for a local endpoint that wants none.
    """
    path = os.environ.get(KEY_FILE_ENV)
    if not path:
        return None
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        logger.debug("rerank key file unreadable: %s", path, exc_info=True)
        return None


def _prompt(query: str, document: str) -> str:
    return (
        f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {_INSTRUCT}\n"
        f"<Query>: {query}\n<Document>: {document}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _post(query: str, document: str, url: str, model: str, key: str | None) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": _prompt(query, document),
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 20,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.load(resp)


def _score_one(query: str, document: str, url: str, model: str, key: str | None) -> float:
    """Score one pair, shrinking the document if the server's window is smaller.

    The context window is a LOAD-TIME choice on the far side, not a property of
    the model: this reranker natively carries 40,960 tokens but was served with
    512, and a 2,000-character drawer is ~650. Seven of thirty candidates came
    back HTTP 400 and, because a partial pool is unusable, the whole rerank was
    discarded — after paying for all thirty requests.

    So a context rejection halves the document and retries rather than failing.
    A truncated judgement is worse than a full one but far better than none, and
    the alternative (capping every document at the smallest window anyone might
    configure) would throw away context that is there on a well-configured
    endpoint. Raise the window on the server for the real fix; this only stops a
    misconfiguration from silently costing the whole second stage.
    """
    text = document[:MAX_TEXT_CHARS]
    for _ in range(MAX_SHRINKS):
        try:
            payload = _post(query, text, url, model, key)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 400 or len(text) <= MIN_TEXT_CHARS:
                raise
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 — the body is a diagnostic, not a dependency
                pass
            if "context" not in detail.lower():
                raise
            text = text[: max(MIN_TEXT_CHARS, len(text) // 2)]
            logger.debug("cross-rerank: context rejected, retrying at %d chars", len(text))
    else:
        raise RuntimeError("cross-rerank: document still too long after shrinking")

    tops = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {t["token"]: t["logprob"] for t in tops}
    yes = max((lp[k] for k in _YES if k in lp), default=_ABSENT)
    no = max((lp[k] for k in _NO if k in lp), default=_ABSENT)
    ey, en = math.exp(yes), math.exp(no)
    return ey / (ey + en) if (ey + en) else 0.0


def score_pairs(query: str, documents: list) -> list | None:
    """Relevance in [0, 1] for each document against ``query``.

    Returns None on any failure — a partial result is not usable, because the
    caller normalises across the whole pool and a missing score would silently
    read as "least relevant" rather than "unknown".
    """
    url = os.environ.get(URL_ENV)
    model = os.environ.get(MODEL_ENV)
    if not url or not model or not query or not documents:
        return None
    key = _api_key()
    try:
        with futures.ThreadPoolExecutor(min(MAX_WORKERS, len(documents))) as pool:
            return list(pool.map(lambda d: _score_one(query, d, url, model, key), documents))
    except Exception:
        logger.debug("cross-rerank scoring failed (non-fatal)", exc_info=True)
        return None
