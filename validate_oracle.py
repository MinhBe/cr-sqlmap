"""
validate_oracle.py - confirm the offline libinjection oracle agrees with the LIVE WAF.

For each payload we compare:
  * oracle  : libinj_oracle.verdict(p)['sqli']  (True = libinjection flags it)
  * live WAF: HTTP status through modsec at 127.0.0.1:8080  (403 = blocked, 200 = passed)

The lab WAF blocks SQLi almost entirely via rule 942100 (libinjection, +5 = threshold),
so the expected correspondence is: oracle.sqli==True  <->  WAF 403.
A payload where oracle.sqli==False but WAF still 403 is driven by a *non*-942100 CRS
regex rule (reported separately). Run on Kali (needs curl + the built .so).

Usage:  python3 validate_oracle.py
"""
import subprocess
import sys

import libinj_oracle

BASE = "http://127.0.0.1:8080"
EP = BASE + "/vulnerabilities/sqli/"

# (payload, note). Mix of known-pass (comment-term / benign) and known-block (extraction,
# boolean, operator) plus tamper-style obfuscations.
BATTERY = [
    "1",
    "1' #",
    "1' -- -",
    "2' #",
    "1'",
    "-1' union select user,password from users#",
    "-1' UNION/**/SELECT/**/user,password/**/FROM/**/users-- -",
    "1' and extractvalue(1,concat(0x7e,(select user from users limit 1)))#",
    "1' AND SLEEP(3)-- -",
    "1'-'0",
    "1' or '1'='1",
    "1' OR 1=1-- -",
    "0' or 1=1 limit 1#",
    "-1'/*!50000union*//*!50000select*/user,password/*!50000from*/users#",
    "1' RLIKE SLEEP(3)-- -",
    "1' AnD sLeEp(3)-- -",
]


def login_cookie():
    """Authenticate to DVWA through the WAF, return the cookie string."""
    jar = "/tmp/vo_cookies.txt"
    subprocess.run(["curl", "-s", "-c", jar, BASE + "/login.php", "-o", "/tmp/vo_login.html"],
                   check=True)
    tok = subprocess.run(
        ["bash", "-c",
         r"""grep -oP "name='user_token'\s+value='\K[a-f0-9]+" /tmp/vo_login.html"""],
        capture_output=True, text=True).stdout.strip()
    subprocess.run(
        ["curl", "-s", "-b", jar, "-c", jar,
         "--data-urlencode", f"user_token={tok}",
         "--data", "username=admin&password=password&Login=Login",
         BASE + "/login.php", "-o", "/dev/null"], check=True)
    return jar


def waf_code(jar, payload):
    r = subprocess.run(
        ["curl", "-s", "-b", jar, "-b", "security=low", "-o", "/dev/null",
         "-w", "%{http_code}", "-G", EP,
         "--data-urlencode", f"id={payload}", "--data-urlencode", "Submit=Submit"],
        capture_output=True, text=True)
    return r.stdout.strip()


def main():
    jar = login_cookie()
    agree = disagree = 0
    rows = []
    for p in BATTERY:
        v = libinj_oracle.verdict(p)
        code = waf_code(jar, p)
        waf_block = (code == "403")
        # 942100-consistent: oracle flags SQLi <=> WAF blocks
        consistent = (v["sqli"] == waf_block)
        agree += consistent
        disagree += (not consistent)
        rows.append((v["sqli"], v["fingerprint"], code, "OK" if consistent else "DIFF", p))
    print(f"{'oracle':6} {'fp':8} {'waf':4} {'match':5}  payload")
    for sqli, fp, code, m, p in rows:
        print(f"{str(sqli):6} {fp:8} {code:4} {m:5}  {p!r}")
    total = agree + disagree
    print(f"\nagreement: {agree}/{total}  ({100*agree/total:.0f}%)  "
          f"[DIFF rows = blocked by a non-942100 CRS regex rule, expected for some]")
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
