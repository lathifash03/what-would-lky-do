# Fetch blocker log — National Archives of Singapore

Recorded: 2026-11-07T06:50Z (real curl runs, see commands below)
User agent used: `what-would-lky-do/0.1 (AI challenge project; educational RAG prototype; +https://github.com/lathifash03/what-would-lky-do)`
Rate: 1 request per 2 seconds. No browser spoofing, no challenge solving attempted.

## Result

| URL | HTTP | Header |
|---|---|---|
| https://www.nas.gov.sg/robots.txt | 202 | `x-amzn-waf-action: challenge` |
| https://www.nas.gov.sg/archivesonline/ | 202 | `x-amzn-waf-action: challenge` |
| https://www.nas.gov.sg/archivesonline/speeches/ | 202 | `x-amzn-waf-action: challenge` |
| https://api.nas.gov.sg/ | 000 | connection failed / no response |

Response headers on the challenged URLs:

```
HTTP/2 202
server: CloudFront
content-length: 0
x-amzn-waf-action: challenge
x-amz-cf-pop: SIN2-P3
```

## Interpretation

nas.gov.sg sits behind AWS WAF Bot Control. Every request from a non-browser client,
including the request for `robots.txt` itself, is answered with an empty HTTP 202 and
`x-amzn-waf-action: challenge`. Solving that challenge requires executing the
AWS WAF JavaScript challenge in a browser context.

`robots.txt` could not be read at all, so the crawl rules are unknown. Fetching under
an unknown robots policy, while defeating an explicit bot challenge, is exactly the
behaviour the project rules forbid.

## Decision

Automated collection from nas.gov.sg is NOT attempted. No challenge-solving,
no browser user-agent spoofing, no headless-browser workaround.
