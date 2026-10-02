from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEMO = Path("demo")
OUT = Path("figures")
SHOTS = Path("data/processing/demo_shots")      # 중간 산출물 (scripts/ → data/processing/)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT_HTTP, PORT_CDP = 8899, 9455
SHOT_W, SHOT_H, SCALE = 1280, 760, 2           # 2배 스케일 → 2560×1520 실픽셀
READY_TIMEOUT = 25.0

# 09시 = 태양 방위 99°(동) · 16시 = 260°(서). 그림자가 정반대로 뻗는 한 쌍이다.
# 16시는 격자 지도 대표 시각이기도 하다 (ADR-008 · `make_figures.MAP_HOUR`).
HOURS = (9, 16)
MODE = "dev"

# 시점 — 전체를 보여주는 시점(span 1620)으로 찍으면 **그림자 차이가 안 읽힌다**.
#   이 그림의 논지가 "시각에 따라 그림자가 이동한다"인데 그게 그림에서 사라지면
#   그림 4·5와 내용만 겹치고 자리만 먹는다.
#
#   그래서 **100m 이상 고층 8동이 모인 곳**(실측 중심 −134, +154 m)에 맞춘다.
#   163m 건물의 그림자가 09시 198m / 16시 196m다. bearing 0(북쪽 위)이라
#   09시는 왼쪽, 16시는 오른쪽으로 뻗어 **좌우 병치에서 대비가 가장 크다.**
#
#   span 780 · pitch 40은 **너무 붙는다** — 타워 협곡 안으로 들어가 그림자가
#      놓일 땅이 안 보인다. 기울일수록 가까운 쪽이 원근으로 커지기 때문이다.
#      물러나고(1150) 눕혀야(30) 지면의 그림자 띠가 읽힌다. 세 조합을 렌더해
#      눈으로 비교한 결과다 (`issue.md` §5-9).
SPAN, PITCH, SHIFT_N, SHIFT_E = 1150.0, 30.0, 20.0, -120.0


# ─────────────────────────────────────────── 최소 WebSocket (CDP 전용)

class WS:
    def __init__(self, url: str):
        assert url.startswith("ws://"), url
        rest = url[len("ws://"):]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port) or 80), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            self._fill()
        head, _, self.buf = self.buf.partition(b"\r\n\r\n")
        if b"101" not in head.split(b"\r\n")[0]:
            raise RuntimeError(f"WebSocket 업그레이드 실패: {head[:200]!r}")

    def _fill(self) -> None:
        d = self.sock.recv(1 << 16)
        if not d:
            raise ConnectionError("CDP 연결이 끊겼다")
        self.buf += d

    def _take(self, n: int) -> bytes:
        while len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text: str) -> None:
        p = text.encode()
        n = len(p)
        hdr = b"\x81"                                   # FIN + text
        mask = os.urandom(4)
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < (1 << 16):
            hdr += b"\xfe" + struct.pack(">H", n)
        else:
            hdr += b"\xff" + struct.pack(">Q", n)
        self.sock.sendall(hdr + mask
                          + bytes(b ^ mask[i % 4] for i, b in enumerate(p)))

    def recv(self) -> str:
        chunks: list[bytes] = []
        while True:
            b0, b1 = self._take(2)
            fin, opcode, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._take(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._take(8))[0]
            if b1 & 0x80:                               # 서버는 마스킹하지 않는다
                self._take(4)
            payload = self._take(n)
            if opcode == 0x9:                           # ping → pong
                self.sock.sendall(b"\x8a\x80" + os.urandom(4))
                continue
            if opcode == 0x8:
                raise ConnectionError("CDP가 연결을 닫았다")
            chunks.append(payload)
            if fin:
                return b"".join(chunks).decode()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class CDP:
    def __init__(self, ws_url: str):
        self.ws, self.n = WS(ws_url), 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:                 # 이벤트는 흘려보낸다
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr,
                      returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"페이지 예외: {r['exceptionDetails'].get('text')}")
        return r.get("result", {}).get("value")


# ─────────────────────────────────────────── 서버 · 브라우저

def serving() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT_HTTP), timeout=0.5):
            return True
    except OSError:
        return False


def start_server() -> subprocess.Popen | None:
    if serving():
        print(f"  정적 서버 이미 떠 있음 :{PORT_HTTP}")
        return None
    p = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT_HTTP),
                          "--directory", str(DEMO)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if serving():
            print(f"  정적 서버 기동 :{PORT_HTTP} (pid {p.pid})")
            return p
        time.sleep(0.25)
    p.kill()
    sys.exit(f"[error] 정적 서버가 안 떴다 :{PORT_HTTP}")


def start_chrome(profile: Path) -> tuple[subprocess.Popen, str]:
    # --force-device-scale-factor로 2배 렌더한다. 캔버스가 2배로 잡히므로
    #   deck.gl이 실제로 고해상도로 그린다 — 캡처 후 확대와 다르다.
    p = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={PORT_CDP}",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
        f"--window-size={SHOT_W},{SHOT_H}", f"--force-device-scale-factor={SCALE}",
        "--hide-scrollbars", "--use-gl=angle", "--use-angle=metal",
        "--enable-unsafe-swiftshader", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT_CDP}/json/list", timeout=1) as r:
                tabs = json.load(r)
            page = next(t for t in tabs if t["type"] == "page")
            return p, page["webSocketDebuggerUrl"]
        except (urllib.error.URLError, OSError, StopIteration, json.JSONDecodeError):
            time.sleep(0.25)
    p.kill()
    sys.exit("[error] 헤드리스 크롬이 안 떴다")


# ─────────────────────────────────────────── 캡처

READY_JS = """(() => {
  if (typeof deckgl === 'undefined' || !deckgl || typeof D === 'undefined' || !D) return 'no-deck';
  if (document.getElementById('loading')) return 'loading';
  const c = document.querySelector('#map canvas');
  if (!c || !c.width) return 'no-canvas';
  if (!deckgl.props.layers || deckgl.props.layers.length < 4) return 'no-layers';
  return 'ready:' + c.width + 'x' + c.height + ':' + hour + ':' + mode;
})()"""


def capture(cdp: CDP, url: str, dest: Path) -> str:
    cdp.call("Page.navigate", url=url)
    t0, state = time.time(), ""
    while time.time() - t0 < READY_TIMEOUT:
        try:
            state = cdp.js(READY_JS) or ""
        except RuntimeError:
            state = "eval-fail"
        if state.startswith("ready:"):
            break
        time.sleep(0.3)
    else:
        sys.exit(f"[error] 렌더 대기 실패 ({state}) — {url}")

    # 렌더 완료 신호 뒤에도 한 박자 기다린다. 그림자 맵은 조명 effect가
    #   적용된 **다음 프레임**에 그려져서, 바로 찍으면 그림자 없는 컷이 나온다.
    cdp.js("new Promise(r => requestAnimationFrame(() => "
           "requestAnimationFrame(() => setTimeout(r, 400))))")

    shot = cdp.call("Page.captureScreenshot", format="png",
                    captureBeyondViewport=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(shot["data"]))
    return state


def verify(path: Path, hour: int) -> dict:
    import matplotlib.image as mpimg
    import numpy as np

    a = mpimg.imread(path)[:, :, :3]
    h, w, _ = a.shape
    # 지도 영역만 본다 — 상하단 UI 패널을 뺀다
    core = a[int(h * .12):int(h * .86), int(w * .05):int(w * .95)]
    bg = np.array([0.051, 0.059, 0.075])                 # index.html clearColor
    far = (np.abs(core - bg).sum(2) > 0.12).mean()       # 배경색이 아닌 픽셀 비율
    return {"px": f"{w}×{h}", "지도픽셀": far,
            "고유색": len(np.unique((core * 255).astype("uint8")
                                    .reshape(-1, 3), axis=0))}


# ─────────────────────────────────────────── 합성

def compose(shots: list[tuple[int, Path]], day: str, meta: dict,
            pitch: float, dpi: float, scale: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0, str(Path.cwd()))
    from core.metrics.make_figures import save, setup      # 폰트·dpi·압축 규약 공유

    setup()
    fig, axes = plt.subplots(1, len(shots), figsize=(13.2, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (hour, p) in zip(axes, shots):
        ax.imshow(mpimg.imread(p))
        ax.set_axis_off()
        s = meta[hour]
        shadow_az = (s["az"] + 180) % 360
        ax.set_title(f"{hour:02d}시 — 태양 방위 {s['az']:.0f}° · 고도 {s['el']:.0f}°"
                     f"   (그림자 → {shadow_az:.0f}°)",
                     fontsize=10.5, pad=6)

        # 그림자 방향을 화살표로 명시한다. 두 컷의 차이가 이 그림의 논지인데
        #   심사위원이 짧게 보는 그림에서 "어느 쪽이 그림자인가"가 애매하면 안 된다.
        #   bearing 0(북쪽 위)이므로 지상 방위 θ의 화면 방향은
        #     dx = sin θ,  dy = −cos θ · cos(pitch)   (화면 y는 아래가 +,
        #   기울인 만큼 남북이 cos(pitch)로 눌린다)
        th = np.radians(shadow_az)
        dx, dy = np.sin(th), -np.cos(th) * np.cos(np.radians(pitch))
        n = np.hypot(dx, dy)
        dx, dy, L = dx / n, dy / n, 0.17
        ax.annotate("", xy=(0.5 + dx*L/2, 0.72 + dy*L/2),
                    xytext=(0.5 - dx*L/2, 0.72 - dy*L/2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#f0a800",
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=22))
        ax.text(0.5, 0.72 - 0.055, "그림자가 뻗는 쪽",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#f0a800", fontweight="bold")
    fig.suptitle(
        f"3D 디지털 트윈 데모 — 같은 골목, 시각만 바꿨다 (관악구 600m · {day})",
        fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.035,
             "슬라이더로 24시각을 넘기면 건물 그림자가 실제 태양 궤적(pvlib NREL SPA)을 따라 이동하고, "
             "그늘에 든 골목이 식는 것이 격자 색으로 따라 바뀐다.\n"
             f"격자 색 = {scale['label']} "
             f"{scale['lo']:+.1f} ~ {scale['hi']:+.1f}{scale['unit']} "
             f"(범위 밖 {scale['clamp']:.1f}%는 클램프) · "
             f"25m 옥외 {scale['cells']:,}칸 · 건물 내부 {scale['inb']}칸 제외 · "
             "인터넷 없이 로컬에서 시연한다.",
             ha="center", va="top", fontsize=9, color="#444")
    fig.tight_layout()
    # dpi를 낮춘다 — 스크린샷 합성은 건물·격자가 섞인 **고주파 래스터**라
    #   무손실 압축이 거의 안 먹는다 (fig4에서 겪었다, `issue.md` §5-8).
    #   원본이 패널당 2560px이라 dpi 110에서도 인쇄 폭 170mm 기준 200dpi를 넘는다.
    save(fig, f"fig10_demo_{day}", dpi=dpi)
    return OUT / f"fig10_demo_{day}.png"


# ─────────────────────────────────────────── main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, nargs="+", default=list(HOURS))
    ap.add_argument("--mode", default=MODE, choices=["dev", "t", "hot", "unc"])
    ap.add_argument("--span", type=float, default=SPAN)
    ap.add_argument("--pitch", type=float, default=PITCH)
    ap.add_argument("--shift", type=float, default=SHIFT_N)
    ap.add_argument("--shift-e", type=float, default=SHIFT_E)
    ap.add_argument("--dpi", type=float, default=110.0)
    ap.add_argument("--keep-shots", action="store_true", help="원본 컷을 지우지 않는다")
    a = ap.parse_args()

    if not Path(CHROME).exists():
        sys.exit(f"[error] 크롬이 없다: {CHROME}")
    idx_p = DEMO / "data" / "days.json"
    for p in (DEMO / "index.html", idx_p):
        if not p.exists():
            sys.exit(f"[error] 없음: {p} — 먼저 `uv run python scripts/build_demo.py`")
    # 그림 10은 **폭염일**로 찍는다 — 그림자 대비가 주제이므로 primary를 쓴다
    idx = json.loads(idx_p.read_text())
    day = idx.get("primary") or idx["days"][0]["key"]
    dist = DEMO / "data" / "district" / f"{day}.json"
    if not dist.exists():
        sys.exit(f"[error] 없음: {dist}")
    sun = json.loads(dist.read_text())["sun"]

    print(f"[캡처] {day} · 시각 {a.hours} · 모드 {a.mode} · "
          f"{SHOT_W}×{SHOT_H} ×{SCALE}")
    srv = start_server()
    profile = Path(f"/tmp/capture_demo_{os.getpid()}")
    chrome, ws_url = start_chrome(profile)
    cdp = None
    try:
        cdp = CDP(ws_url)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        shots = []
        for h in a.hours:
            dest = SHOTS / f"demo_{day}_{h:02d}h_{a.mode}.png"
            url = (f"http://localhost:{PORT_HTTP}/?hour={h}&mode={a.mode}"
                   f"&span={a.span}&pitch={a.pitch}&bearing=0"
                   f"&shift={a.shift}&shiftE={a.shift_e}")
            state = capture(cdp, url, dest)
            v = verify(dest, h)
            print(f"  {h:02d}시 → {dest.name}  {v['px']} · "
                  f"지도픽셀 {v['지도픽셀']*100:.1f}% · 고유색 {v['고유색']:,} · {state}")
            if v["지도픽셀"] < 0.5:
                sys.exit(f"[error] {h}시 컷이 사실상 비었다 "
                         f"(지도픽셀 {v['지도픽셀']*100:.1f}%) — 렌더 실패")
            shots.append((h, dest))
        # 색 범위를 캡션에 하드코딩하지 않는다 — 페이지가 데이터에서 계산한
        #   값을 그대로 읽어 온다. `build_demo.py`를 다시 돌려 값이 바뀌면
        #   캡션도 따라 바뀐다 (신청서 수치가 코드와 어긋나면 안 된다).
        scale = json.loads(cdp.js(
            "JSON.stringify({lo:RANGE[mode][0], hi:RANGE[mode][1], "
            "clamp:MODES[mode].clamp, label:MODES[mode].label, unit:MODES[mode].unit, "
            "cells:OUT.length, inb:D.cx.length-OUT.length})"))
    finally:
        if cdp:
            cdp.ws.close()
        chrome.terminate()
        chrome.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)
        if srv:
            srv.terminate()

    out = compose(shots, day, {h: sun[h] for h in a.hours}, a.pitch, a.dpi, scale)
    if not a.keep_shots:
        for _, p in shots:
            p.unlink()
        print(f"  원본 컷 삭제 (남기려면 --keep-shots)")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
