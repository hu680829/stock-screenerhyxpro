"""Send the daily screener CSV summary through PushPlus (WeChat notification)."""
from __future__ import annotations

import csv
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

PUSHPLUS_ENDPOINT = "https://www.pushplus.plus/send"


def build_message(csv_path: str | Path) -> tuple[str, str]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ranking report not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Ranking report is empty")

    ranked = sorted(
        rows,
        key=lambda row: int(row.get("rank") or 10**9),
    )
    eligible = [row for row in ranked if str(row.get("risk_pass", "")).lower() == "true"]
    top = ranked[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"每日量化选股报告｜通过风控 {len(eligible)} 只"
    lines = [
        "<h2>每日多因子选股报告</h2>",
        f"<p>生成时间：{html.escape(stamp)}</p>",
        f"<p>扫描标的：{len(ranked)} 只；通过风控：{len(eligible)} 只。</p>",
        "<p>注意：这是研究筛选结果，不是买卖指令或收益保证。</p>",
        "<table border='1' cellpadding='6' cellspacing='0'>",
        "<tr><th>排名</th><th>代码</th><th>综合分</th><th>风控</th><th>结果</th></tr>",
    ]
    for row in top:
        passed = str(row.get("risk_pass", "")).lower() == "true"
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('rank', '')))}</td>"
            f"<td>{html.escape(str(row.get('ticker', '')))}</td>"
            f"<td>{html.escape(str(row.get('composite_score', 'N/A')))}</td>"
            f"<td>{'通过' if passed else '未通过'}</td>"
            f"<td>{html.escape(str(row.get('recommendation', '')))}</td>"
            "</tr>"
        )
    lines.append("</table>")
    if not eligible:
        lines.append("<p><b>风险提示：</b>本次没有标的通过风控条件，请勿为了交易而交易。</p>")
    lines.append("<p>完整 CSV 请在 GitHub Actions 的 quant-rankings 工件中查看。</p>")
    return title, "\n".join(lines)


def append_portfolio_snapshot(message: str, snapshot_path: str | Path = "reports/portfolio_snapshot.json") -> str:
    """Append a portfolio status section if the monitor produced a report."""
    path = Path(snapshot_path)
    if not path.is_file():
        return message
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return message + "\\n<p><b>持仓监控：</b>报告无法读取，请检查 GitHub Actions 日志。</p>"
    if snapshot.get("status") != "ok":
        error = html.escape(str(snapshot.get("error", "数据不可用")))
        return message + f"\\n<h3>持仓风险监控</h3><p><b>数据暂不可用：</b>{error}。本次不计算持仓收益或再平衡建议。</p>"
    lines = [
        "\\n<h3>持仓风险监控（非交易指令）</h3>",
        f"<p>共同交易日：{html.escape(str(snapshot.get('previous_session')))} → {html.escape(str(snapshot.get('latest_session')))}</p>",
        f"<p>按人民币计价组合区间收益：{float(snapshot['portfolio_return'])*100:.2f}%</p>",
        f"<p>USD/CNH 区间变化：{float(snapshot['usd_cnh_return'])*100:.2f}%</p>",
        "<table border='1' cellpadding='6' cellspacing='0'><tr><th>标的</th><th>目标权重</th><th>当前权重</th><th>偏离（百分点）</th></tr>",
    ]
    for ticker, target in snapshot["target_weights"].items():
        current = snapshot["current_weights"][ticker]
        drift = snapshot["drift_pp"][ticker]
        lines.append(
            f"<tr><td>{html.escape(str(ticker))}</td>"
            f"<td>{target*100:.2f}%</td><td>{current*100:.2f}%</td>"
            f"<td>{drift:+.2f}</td></tr>"
        )
    lines.append("</table>")
    alerts = snapshot.get("rebalance_alerts") or {}
    if alerts:
        lines.append("<p><b>权重偏离达到再平衡死区：</b>" +
                     html.escape(", ".join(f"{k} ({v:+.2f}pp)" for k, v in alerts.items())) +
                     "。请人工核验后再决定是否调整。</p>")
    else:
        lines.append("<p>当前没有标的达到再平衡偏离阈值。</p>")
    return message + "\\n" + "\\n".join(lines)


def send_pushplus(token: str, title: str, content: str, timeout: int = 20) -> dict:
    payload = json.dumps({
        "token": token,
        "title": title,
        "content": content,
        "template": "html",
    }).encode("utf-8")
    request = Request(
        PUSHPLUS_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "stock-screenerhyxpro/1.0"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"PushPlus request failed: {type(exc).__name__}") from exc
    if not 200 <= status < 300:
        raise RuntimeError(f"PushPlus returned HTTP {status}")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PushPlus returned non-JSON response") from exc
    # PushPlus commonly returns code 200 on acceptance; fail loudly on API rejection.
    if str(result.get("code")) != "200":
        raise RuntimeError(f"PushPlus rejected notification (code={result.get('code')})")
    return result


def main() -> int:
    token = os.environ.get("PUSHPLUS_TOKEN", "").strip()
    if not token:
        print("ERROR: PUSHPLUS_TOKEN is not configured in GitHub Secrets.", file=sys.stderr)
        return 2
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "reports/quant_rankings.csv"
    try:
        title, content = build_message(csv_path)
        result = send_pushplus(token, title, content)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: notification failed: {exc}", file=sys.stderr)
        return 1
    print(f"PushPlus accepted notification (code={result.get('code')}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
