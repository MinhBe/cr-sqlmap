"""
libinj_oracle.py - offline libinjection oracle (CR-sqlmap Phase 1, "libinjection layer").

Wraps the libinjection C core -- the SAME engine ModSecurity CRS uses as rule 942100
(`@detectSQLi`) -- via ctypes, so a candidate payload's SQLi verdict + 5-char fingerprint
is computed LOCALLY with no WAF round-trip. This is the dense signal that fixes the
reward-is-zero problem: a payload whose folded token fingerprint is NOT in libinjection's
bad-set is a candidate to slip rule 942100 (anomaly +5 -> 403), even though it is a real,
executable injection.

Build the shared lib once (on Kali, matching the WAF's ModSecurity 3.0.17):
    git clone --depth 1 https://github.com/libinjection/libinjection
    gcc -std=c99 -O2 -fPIC -shared -Isrc -o <dir>/liblibinjection.so \
        src/libinjection_sqli.c src/libinjection_xss.c src/libinjection_html5.c
Point LIBINJECTION_SO at the result (default /home/kali/bin/liblibinjection.so).

Verdicts validated 1:1 against the live WAF (modsec 3.0.17 / CRS 1.30.6) on the lab
battery (2026-10-23): 1' # / 1' -- - -> sqli=0 (WAF 200); union/boolean/operator -> sqli=1
(WAF 403). See validate_oracle.py for the reproducible check.
"""
import ctypes
import os
import re
import threading

_SO_PATH = os.environ.get("LIBINJECTION_SO", "/home/kali/bin/liblibinjection.so")
_lib = None
_lock = threading.Lock()

# ModSecurity CRS rule 942100 (@detectSQLi) applies these transformations to the
# argument BEFORE handing it to libinjection:  t:none,t:utf8toUnicode,t:urlDecodeUni,
# t:removeNulls  (read from the live container, 2026-10-23). To predict the WAF's
# verdict we must run libinjection on the TRANSFORMED string, not the raw request
# bytes -- otherwise every %XX/%uXXXX encoding tamper looks like a bypass to the raw
# oracle while modsec decodes it right back and still blocks. verdict(modsec=True)
# (the default) replays these transforms; modsec=False inspects the raw string.
_RE_UNI = re.compile(r"%u([0-9a-fA-F]{4})")
_RE_HEX = re.compile(r"%([0-9a-fA-F]{2})")


def transform_like_modsec(s):
    """Replay rule 942100's urlDecodeUni + removeNulls (+ best-effort utf8toUnicode)."""
    s = _RE_UNI.sub(lambda m: chr(int(m.group(1), 16)), s)
    s = _RE_HEX.sub(lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace("\x00", "")  # removeNulls
    # utf8toUnicode: collapse any UTF-8 byte sequence (incl. overlong) to its codepoint.
    try:
        s = s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s


def _load():
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                lib = ctypes.CDLL(_SO_PATH)
                # int libinjection_sqli(const char *s, size_t slen, char fingerprint[]);
                lib.libinjection_sqli.restype = ctypes.c_int
                lib.libinjection_sqli.argtypes = [
                    ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                ]
                _lib = lib
    return _lib


def verdict(payload, modsec=True):
    """Return {'sqli': bool, 'fingerprint': str}.

    sqli=True  -> libinjection fingerprints it as SQLi: CRS rule 942100 fires (+5) -> block.
    sqli=False -> fingerprint NOT in the bad-set -> candidate to slip rule 942100.

    modsec=True (default) first replays rule 942100's transforms (transform_like_modsec)
    so the verdict predicts the LIVE WAF on encoded payloads; modsec=False inspects the
    raw string as given. `payload` may be str or bytes; an explicit length is passed to C,
    so embedded NULs and newline/tab tamper output are handled correctly.
    """
    if isinstance(payload, (bytes, bytearray)):
        s = bytes(payload).decode("latin1")
    else:
        s = str(payload)
    if modsec:
        s = transform_like_modsec(s)
    b = s.encode("utf-8", "surrogatepass")
    fp = ctypes.create_string_buffer(16)
    r = _load().libinjection_sqli(b, len(b), fp)
    return {"sqli": bool(r), "fingerprint": fp.value.decode("latin1")}


def is_bypass_candidate(payload, modsec=True):
    """True iff libinjection does NOT flag the payload (fingerprint left the bad-set)."""
    return not verdict(payload, modsec=modsec)["sqli"]


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    lines = args if args else [ln.rstrip("\n") for ln in sys.stdin]
    for ln in lines:
        v = verdict(ln)
        print(f"sqli={int(v['sqli'])}\tfp={v['fingerprint']}\t{ln!r}")
