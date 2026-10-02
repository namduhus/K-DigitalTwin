from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

OUT = Path("figures")
TRAIN = Path("data/processing/train.parquet")
HORIZON = Path("data/processing/sensor_horizon.parquet")
LOCATIONS = Path("data/processing/sdot_locations.csv")
GRID = Path("data/processing/grid_features.parquet")
BOUNDARY = Path("data/processing/seoul_boundary.parquet")
PRED = {"2026-08-06": Path("data/metric/grid_pred_2026-08-06.parquet"),
        "2026-07-15": Path("data/metric/grid_pred_2026-07-15.parquet")}

HOT_C, TROPICAL_C = 33.0, 25.0
NIGHT_START_H, NIGHT_LEN_H = 18, 15      # 열대야 구간 18:01~다음날 09:00

# 남중 12:38 KST — 한국 표준자오선 135°E vs 서울 127°E 때문이다. `pvlib`가 자동 처리한다.
SOLAR_NOON = 12 + 38 / 60
AZ_STEP, N_AZ = 2.0, 180

# 격자 지도의 대표 시각 — 하드코딩하지 않는다 (8/25 변경: 14 → 16)
#
# 8/24까지 fig4·fig5가 14시였다. `core/metrics/shadow_probe.py`(검증 1)에서
# **14시는 25m 격자에 그림자가 사실상 없는 시각**임이 드러났다 — 태양고도 61°에서
# 그늘 셀 0.4%, 그늘·양지가 섞인 400m 창이 서울 전체에 18개뿐이다. 고도 68°인
# 정오에는 0.1%까지 떨어진다. 20m 건물의 정오 그림자가 8m로 **셀(25m)보다 짧아**
# 중심점 판정에서 사라지기 때문이고, 이건 버그가 아니라 해상도 한계다.
#
# 16시(고도 39.8°)는 그늘 셀 5.2%이면서 창 안 Δt(그늘−양지)가 **−0.137℃**로
# 유지되는 자리다. 더 늦추면 그림자는 길어지지만 **고도 16° 아래에서 Δt 부호가
# +로 뒤집혀**(18시 +0.064 · 19시 +0.178) 지도가 주장을 배신한다.
#
# 골목 확대 그림과 전역 지도의 시각을 **하나로 통일**한다 — 사다리(5km → 25m →
# 5km 한 칸 → 골목)가 같은 시각이어야 연속으로 읽힌다.
MAP_HOUR = 16


def setup() -> None:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for want in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Nanum Gothic"):
        if want in have:
            plt.rcParams["font.family"] = want
            break
    else:
        print("한글 폰트를 못 찾았다 — 라벨이 깨진다", file=sys.stderr)
    plt.rcParams.update({
        "axes.unicode_minus": False,      # 한글 폰트에 U+2212가 없어 □로 나온다
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str, dpi: float | None = None) -> None:
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=dpi,
                pil_kwargs={"optimize": True, "compress_level": 9})
    plt.close(fig)
    print(f"  → {p} ({p.stat().st_size/1e3:.0f} KB)")


# ──────────────────────────────────────────────── 그림 2

def fig2() -> None:
    d = pd.read_parquet(TRAIN, columns=["ts", "tp", "illu", "sun_el", "is_shadow",
                                        "is_night", "sensor_ok"])
    d = d[d.sensor_ok & d.tp.notna()]
    day = d[~d.is_night & d.is_shadow.notna()].copy()
    day["hour"] = day.ts.dt.hour
    day["sh"] = day.is_shadow.astype(bool)

    byh = day.groupby("hour")["sh"].mean() * 100
    med = d.groupby("ts")["tp"].transform("median")
    day["anom"] = day.tp - med.loc[day.index]

    bins = [0, 10, 20, 30, 40, 50, 60, 90]
    day["band"] = pd.cut(day.sun_el, bins)
    rows = []
    for b, g in day.groupby("band", observed=True):
        s, u = g[g.sh], g[~g.sh]
        # 임계값을 100으로 둔다. 500이면 `60~90°`(그늘 479행)가 빠지는데
        # 그 구간이 **조도비 0.13x · 기온차 -1.10℃로 가장 강한 증거**다.
        # 대신 표본 수를 그림에 함께 적어 독자가 판단하게 한다.
        if len(s) < 100 or len(u) < 100:
            continue
        rows.append({"lab": f"{int(b.left)}~{int(b.right)}°",
                     "illu_ratio": s.illu.median() / max(u.illu.median(), 1e-9),
                     "dT": s.anom.median() - u.anom.median(),
                     "n_sh": len(s), "n_sun": len(u)})
    r = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(byh.index, byh.values, "o-", color="#1f4e79", lw=2, ms=5)
    ax1.axvline(SOLAR_NOON, color="#c00", ls="--", lw=1.2)
    ax1.annotate(f"남중 12:38", (SOLAR_NOON, byh.max() * 0.9), color="#c00",
                 fontsize=9, ha="left", xytext=(4, 0), textcoords="offset points")
    ax1.set(xlabel="시각 (KST)", ylabel="그늘 판정 비율 (%)",
            title="① 시각별 그늘 비율 — 최저점이 남중과 일치", xticks=range(4, 21, 2))
    ax1.set_ylim(0, 100)

    x = np.arange(len(r))
    ax2.bar(x - 0.2, r.illu_ratio, 0.4, label="조도비 (그늘/양지)", color="#f0a800")
    ax2b = ax2.twinx()
    ax2b.bar(x + 0.2, r.dT, 0.4, label="기온차 (그늘-양지) ℃", color="#1f4e79")
    ax2b.axhline(0, color="#666", lw=0.8)
    ax2b.grid(False)
    for xi, (rt, ns) in enumerate(zip(r.illu_ratio, r.n_sh)):
        ax2.annotate(f"{rt:.2f}\nn={ns:,}", (xi - 0.2, rt), ha="center", va="bottom",
                     fontsize=7, color="#7a5200")
    ax2.set(xticks=x, xticklabels=r.lab, xlabel="태양 고도", ylabel="조도비 (배)",
            title="② 태양이 높을수록 그늘 효과가 커진다  (n = 그늘 표본)", ylim=(0, 1.30))
    ax2b.set_ylabel("기온차 (℃)")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    fig.suptitle("음영 계산 독립검증도 — 조도는 계산에 쓰지 않은 관측이다",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig2_shadow_validation")
    print(f"     그늘 비율 {byh.min():.1f}%(최저 {byh.idxmin()}시) ~ {byh.max():.1f}%")
    print(f"     조도비 {r.illu_ratio.iloc[-1]:.2f}x(고각) ~ {r.illu_ratio.iloc[0]:.2f}x(저각)")


# ──────────────────────────────────────────────── 그림 3

def fig3() -> None:
    import pvlib
    hz = pd.read_parquet(HORIZON)
    loc = pd.read_csv(LOCATIONS).set_index("sn")
    hzc = [f"hz_{i:03d}" for i in range(N_AZ)]
    lo, hi = hz.loc[hz.svf.idxmin()], hz.loc[hz.svf.idxmax()]

    # 폭염일 태양 궤적 (10분 간격)
    ts = pd.date_range("2026-08-06 04:00", "2026-08-06 20:00", freq="10min",
                       tz="Asia/Seoul")
    sp = pvlib.solarposition.spa_python(ts, 37.5665, 126.9780)
    saz, sel = sp.azimuth.to_numpy(), sp.apparent_elevation.to_numpy()
    ok = sel > 0
    saz, sel, ts = saz[ok], sel[ok], ts[ok]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6),
                             subplot_kw={"projection": "polar"})
    az = np.arange(N_AZ) * AZ_STEP
    for ax, row, tag in zip(axes, (lo, hi), ("밀집지", "개방지")):
        # fisheye — **중심이 천정(90°), 바깥이 지평선(0°)**. 하늘 반구를 위에서 본
        # 어안 투영이 이 분야의 관례다. r = 90 - 고도로 뒤집는다. 이렇게 하면
        # 건물 실루엣이 바깥 테두리에서 안쪽으로 밀려 들어오고 태양 궤적이 호가 되어
        # "하늘이 얼마나 막혔는가"가 직관적으로 읽힌다.
        prof = row[hzc].to_numpy(float)
        th = np.radians(np.r_[az, az[0]])
        rr = 90 - np.r_[prof, prof[0]]
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.fill_between(th, rr, 90, color="#8a8a8a", alpha=0.55, lw=0)   # 막힌 하늘
        ax.plot(th, rr, color="#444", lw=1.0)

        shaded = sel < prof[(saz / AZ_STEP).astype(int) % N_AZ]
        ax.plot(np.radians(saz[~shaded]), 90 - sel[~shaded], ".", ms=4.5,
                color="#f0a800", label="양지", zorder=5)
        ax.plot(np.radians(saz[shaded]), 90 - sel[shaded], ".", ms=4.5,
                color="#1f4e79", label="그늘", zorder=5)
        for h in (6, 9, 12, 15, 18):
            m = (ts.hour == h) & (ts.minute == 0)
            if m.any():
                i = np.where(m)[0][0]
                ax.annotate(f"{h}시", (np.radians(saz[i]), 90 - sel[i]), fontsize=8,
                            color="#222", xytext=(4, 4), textcoords="offset points",
                            zorder=6)
        ax.set_rlim(0, 90)
        ax.set_rticks([0, 30, 60, 90])
        ax.set_yticklabels(["천정 90°", "60°", "30°", "지평선 0°"], fontsize=7.5)
        ax.set_rlabel_position(112)
        gu = loc.gu.get(row.sn, "")
        ax.set_title(f"{tag} · {gu}\nSVF {row.svf:.3f} · 그늘 시간 {shaded.mean()*100:.0f}%"
                     f" · 최근접 건물 {row.bld_d_min:.1f} m", pad=26, fontsize=10)
    axes[0].legend(loc="lower left", bbox_to_anchor=(-0.16, -0.10), fontsize=9)
    fig.subplots_adjust(top=0.78, wspace=0.35)
    fig.suptitle("수평선 프로파일·태양궤적도 (fisheye) — 회색이 건물에 가려진 하늘, "
                 "점이 2026-08-06 태양",
                 fontsize=12, fontweight="bold", y=0.99)
    save(fig, "fig3_horizon_polar")
    print(f"     밀집지 {lo.sn} SVF {lo.svf:.3f} / 개방지 {hi.sn} SVF {hi.svf:.3f}")


# ──────────────────────────────────────────────── 격자 지도 공용

class Raster:
    def __init__(self, cell: float = 25.0):
        g = pd.read_parquet(GRID, columns=["gx", "gy", "in_building"])
        self.gx0, self.gy0 = int(g.gx.min()), int(g.gy.min())
        self.W = int(g.gx.max()) - self.gx0 + 1
        self.H = int(g.gy.max()) - self.gy0 + 1
        self.cell = cell
        self.col = g.gx.to_numpy() - self.gx0
        self.row = g.gy.to_numpy() - self.gy0
        self.key = pd.MultiIndex.from_arrays([g.gx, g.gy])
        self.in_bld = g.in_building.to_numpy().astype(bool)
        # imshow용 extent (km 단위, EPSG:5186)
        self.extent = [(self.gx0 + 0.5) * cell / 1000, (self.gx0 + self.W + 0.5) * cell / 1000,
                       (self.gy0 + 0.5) * cell / 1000, (self.gy0 + self.H + 0.5) * cell / 1000]
        print(f"     래스터 {self.W} × {self.H} (경계 내 {len(g):,}셀 · 건물내부 {self.in_bld.sum():,})")

    def to_array(self, gx, gy, v, mask_building: bool = False) -> np.ndarray:
        a = np.full((self.H, self.W), np.nan, np.float32)
        a[gy - self.gy0, gx - self.gx0] = v
        if mask_building:
            a[self.row[self.in_bld], self.col[self.in_bld]] = np.nan
        return a[::-1]        # imshow는 위에서 아래로 그린다


def _boundary_lines(ax):
    from shapely import from_wkt
    b = pd.read_parquet(BOUNDARY)
    for _, r in b[b.sig_cd != "11000"].iterrows():
        geo = from_wkt(r.wkt)
        for poly in (geo.geoms if geo.geom_type == "MultiPolygon" else [geo]):
            x, y = poly.exterior.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000,
                    color="#333", lw=0.4, alpha=0.65, zorder=3)


def _pick_hour(day: str, hour: int) -> pd.Timestamp:
    import pyarrow.parquet as pq
    ts = pd.DatetimeIndex(pd.unique(pq.read_table(PRED[day], columns=["ts"])["ts"].to_pandas()))
    same = ts[(ts.hour == hour) & (ts.normalize() == pd.Timestamp(day))]
    return same[0] if len(same) else ts[np.abs(ts.hour - hour).argmin()]


def _panel(ax, arr, r: Raster, title: str, cmap: str, label: str,
           vmin=None, vmax=None, cbar: bool = True):
    im = ax.imshow(arr, extent=r.extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=2)
    _boundary_lines(ax)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_facecolor("#f7f7f7")
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.043, pad=0.02)
        cb.ax.set_title(label, fontsize=8.5, pad=6)   # 세로 라벨은 축에 눌려 잘린다
        cb.ax.tick_params(labelsize=8)
    return im


# ──────────────────────────────────────────────── 그림 4

def fig4(day: str = "2026-08-06") -> None:
    import pyarrow.parquet as pq
    r = Raster()
    hh = _pick_hour(day, MAP_HOUR)
    print(f"     대표 시각 {hh:%m-%d %H:%M}")

    # ① 14시 슬라이스 — 필터로 969,212행만 읽는다
    s = pq.read_table(PRED[day], columns=["gx", "gy", "t_hat", "q10", "q90"],
                      filters=[("ts", "=", hh)]).to_pandas()
    a_t = r.to_array(s.gx.to_numpy(), s.gy.to_numpy(), s.t_hat.to_numpy())
    a_w = r.to_array(s.gx.to_numpy(), s.gy.to_numpy(),
                     (s.q90 - s.q10).to_numpy())

    # ② 궤적 파생량 — 전 시각 t_hat만 읽어 접는다
    p = pq.read_table(PRED[day], columns=["gx", "gy", "ts", "t_hat"]).to_pandas()
    hot = (p.assign(v=(p.t_hat >= HOT_C))
             .groupby(["gx", "gy"], sort=False)["v"].sum().reset_index())
    # 열대야 구간은 `night_key`로 묶어야 한다. `(ts-18h).hour < 15`만 쓰면
    # **그날 새벽(00~08시, 전날 밤에 속함)까지 섞여** 24시간이 나온다(실측 버그).
    sh = p.ts - pd.Timedelta(hours=NIGHT_START_H)
    night = p[sh.dt.floor("D") == pd.Timestamp(day)]
    trop = (night.assign(v=(night.t_hat >= TROPICAL_C))
                 .groupby(["gx", "gy"], sort=False)["v"].sum().reset_index())
    # 야간 최저기온 — 열대야 지속시간이 폭염일엔 전역 포화되어 정보가 없다.
    # 연속값인 최저기온이 "밤에 얼마나 안 식는가"를 공간적으로 보여준다.
    tmin = night.groupby(["gx", "gy"], sort=False)["t_hat"].min().reset_index()
    n_ts_night = night.ts.nunique()
    del p, night
    a_hot = r.to_array(hot.gx.to_numpy(), hot.gy.to_numpy(), hot.v.to_numpy(float))
    a_trp = r.to_array(trop.gx.to_numpy(), trop.gy.to_numpy(), trop.v.to_numpy(float))
    a_min = r.to_array(tmin.gx.to_numpy(), tmin.gy.to_numpy(), tmin.t_hat.to_numpy())
    valid = ~np.isnan(a_trp)
    sat = float((a_trp[valid] >= n_ts_night - 0.5).mean())   # NaN 제외. 포함하면 희석된다
    print(f"     열대야 구간 {n_ts_night}시각 · 지속시간 중위 {np.nanmedian(a_trp):.0f}h · "
          f"**최대값 포화 셀 {sat*100:.0f}%**")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4))
    _panel(axes[0, 0], a_t, r, f"① 체감더위 추정 — {hh:%m월 %d일 %H시}",
           "YlOrRd", "기온 (℃)",
           vmin=np.nanpercentile(a_t, 1), vmax=np.nanpercentile(a_t, 99))
    _panel(axes[0, 1], a_hot, r, "② 33℃ 초과 누적 시간 (하루)",
           "YlOrRd", "시간", vmin=0, vmax=np.nanpercentile(a_hot, 99.5))
    _panel(axes[1, 0], a_min, r, "③ 야간 최저기온 (18시~다음날 9시)",
           "YlOrRd", "기온 (℃)",
           vmin=np.nanpercentile(a_min, 1), vmax=np.nanpercentile(a_min, 99))
    _panel(axes[1, 1], a_w, r, "④ 80% 예측구간 폭 (보정 전) — 넓을수록 불확실하다",
           "Purples", "℃", vmin=0, vmax=np.nanpercentile(a_w, 99))
    fig.suptitle(f"25m 격자 4패널도 - 서울 전역 {day} (969,212셀, 건물 내부 셀 포함)",
                 fontsize=13, fontweight="bold", y=0.965)
    fig.subplots_adjust(hspace=0.10, wspace=0.02)
    # 이 그림만 dpi를 낮춘다 (150 → 110). 4패널 × 969,212셀 래스터라 **고주파 잡음이
    # 많아 PNG 압축이 먹지 않는다** — 무손실 압축(`compress_level=9`)을 걸어도 1.2 MB에서
    # 줄지 않았다. 신청서가 「이미지 용량 축소」를 요구하고, A4 반 페이지로 들어갈 때
    # 11.5인치 figsize는 어차피 축소되므로 110dpi로도 인쇄 해상도가 남는다.
    save(fig, f"fig4_grid_{day}", dpi=110)
    for lab, arr in (("기온", a_t), ("33℃ 초과", a_hot), ("야간최저", a_min),
                     ("열대야h", a_trp), ("구간폭", a_w)):
        print(f"     {lab:<9} 중위 {np.nanmedian(arr):>6.2f} · "
              f"P05 {np.nanpercentile(arr,5):>6.2f} · P95 {np.nanpercentile(arr,95):>6.2f}")


# ──────────────────────────────────────────────── 그림 5

def fig5(day: str = "2026-08-06") -> None:
    import pyarrow.parquet as pq
    r = Raster()
    hh = _pick_hour(day, MAP_HOUR)
    CELL5 = 5000.0

    # 25m 예측 + 좌표
    s = pq.read_table(PRED[day], columns=["gx", "gy", "t_hat"],
                      filters=[("ts", "=", hh)]).to_pandas()
    gxy = pd.read_parquet(GRID, columns=["gx", "gy", "x", "y"])
    s = s.merge(gxy, on=["gx", "gy"], how="left")
    s["c5x"] = np.floor(s.x / CELL5).astype(int)
    s["c5y"] = np.floor(s.y / CELL5).astype(int)

    # 5km 값 — 그 칸 안 센서 관측의 중위 (센서 없으면 서울 전체 중위)
    obs = pd.read_parquet(TRAIN, columns=["sn", "ts", "tp", "sensor_ok"])
    obs = obs[(obs.ts == hh) & obs.sensor_ok & obs.tp.notna()]
    loc = pd.read_csv(LOCATIONS).set_index("sn")
    obs = obs.join(loc[["x", "y"]], on="sn")
    obs["c5x"] = np.floor(obs.x / CELL5).astype(int)
    obs["c5y"] = np.floor(obs.y / CELL5).astype(int)
    per5 = obs.groupby(["c5x", "c5y"])["tp"].agg(["median", "size"])
    seoul_med = float(obs.tp.median())
    s = s.join(per5["median"].rename("v5"), on=["c5x", "c5y"])
    n_fill = int(s.v5.isna().sum())
    s["v5"] = s.v5.fillna(seoul_med)
    n5 = s.groupby(["c5x", "c5y"]).ngroups
    print(f"     5km 칸 **{n5}개** (센서 보유 {len(per5)}개 · 센서 없는 칸 대체 {n_fill/len(s)*100:.1f}% 셀)")
    print(f"     서울 전체 중위 {seoul_med:.2f}℃ · 5km 값 범위 {s.v5.min():.2f}~{s.v5.max():.2f}℃")

    a25 = r.to_array(s.gx.to_numpy(), s.gy.to_numpy(), s.t_hat.to_numpy())
    a5 = r.to_array(s.gx.to_numpy(), s.gy.to_numpy(), s.v5.to_numpy())
    vmin, vmax = np.nanpercentile(a25, 1), np.nanpercentile(a25, 99)

    # 확대할 칸 — 25m 내부 변동이 가장 큰 5km 칸
    #
    # **서울 경계 안에 온전히 든 칸만 후보다.** 임계값이 5,000셀이던 때
    # 경계에 걸친 칸(27,497셀)이 뽑혔다 — 산지와 시가지가 한 칸에 섞여 표준편차가
    # 커지기 때문이다. 그러면 ③ 패널의 오른쪽이 비고 **"1칸 = 40,000칸"이라는
    # 제목과 셀 수가 어긋난다.** 그림의 결정타가 정확히 그 등식이므로 깨면 안 된다.
    FULL = int((CELL5 / 25.0) ** 2)            # 5000² / 25² = 40,000
    var = s.groupby(["c5x", "c5y"])["t_hat"].agg(["std", "min", "max", "size"])
    var = var[var["size"] >= FULL * 0.98]
    print(f"     온전한 5km 칸 {len(var)}개 / 전체 {n5}개 (경계 걸침 제외)")
    cx, cy = var["std"].idxmax()
    z = s[(s.c5x == cx) & (s.c5y == cy)]
    z5 = float(z.v5.iloc[0])
    print(f"     확대 칸 ({cx},{cy}) · 25m 셀 {len(z):,}개 · "
          f"5km 값 {z5:.2f}℃ vs 25m {z.t_hat.min():.2f}~{z.t_hat.max():.2f}℃")

    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.82], wspace=0.06)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    _panel(ax1, a5, r, f"① 기상청 5km 해상도 — 서울 전체 {n5}칸", "YlOrRd",
           "기온 (℃)", vmin=vmin, vmax=vmax, cbar=False)
    _panel(ax2, a25, r, f"② 우리 추정 25m — 969,212칸", "YlOrRd",
           "기온 (℃)", vmin=vmin, vmax=vmax)
    # 확대 칸 위치 표시
    for ax in (ax1, ax2):
        ax.add_patch(plt.Rectangle((cx * CELL5 / 1000, cy * CELL5 / 1000),
                                   CELL5 / 1000, CELL5 / 1000, fill=False,
                                   ec="#0b3d91", lw=1.8, zorder=6))

    # ③ 확대 — 5km 한 칸 안의 25m
    zi = r.to_array(z.gx.to_numpy(), z.gy.to_numpy(), z.t_hat.to_numpy())
    x0, x1 = cx * CELL5 / 1000, (cx + 1) * CELL5 / 1000
    y0, y1 = cy * CELL5 / 1000, (cy + 1) * CELL5 / 1000
    im = ax3.imshow(zi, extent=r.extent, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
    ax3.set_xlim(x0, x1); ax3.set_ylim(y0, y1)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.grid(False)
    ax3.set_facecolor("#f7f7f7")
    for sp in ax3.spines.values():
        sp.set(color="#0b3d91", linewidth=1.8, visible=True)
    # 등식(1칸 = 40,000칸)이 이 그림의 결정타다. 실제 그려진 셀은 서울 경계 안
    # 39,445개라 숫자가 다르므로 **둘을 함께** 적는다 — 어느 한쪽만 쓰면 어긋난다.
    ax3.set_title(f"③ 그 1칸을 열면 — 25m {FULL:,}칸 (서울 경계 내 {len(z):,})",
                  fontsize=10, pad=6)
    # matplotlib은 마크다운을 모른다 — `**`가 문자로 그대로 찍힌다. weight로 강조한다
    ax3.text(0.5, -0.055, f"5km는 이 칸 전체에 {z5:.1f}℃ 하나",
             transform=ax3.transAxes, ha="center", va="top", fontsize=11,
             color="#0b3d91", fontweight="bold")
    # 인용은 분위 기반으로 한다 (`CLAUDE.md` 작업원칙 6). `max−min`은 39,445셀 중
    #   양끝 두 개가 정하는 값이라 조건이 조금만 바뀌어도 흔들린다 — 실제로 대표
    #   시각을 14시→16시로 바꾸자 4.93→4.24로 움직였다. 범위는 보여주되
    #   **숫자로 인용하는 폭은 P95−P05**로 낸다.
    p05, p95 = np.percentile(z.t_hat, [5, 95])
    ax3.text(0.5, -0.125, f"25m는 {z.t_hat.min():.1f} ~ {z.t_hat.max():.1f}℃ — "
                          f"양끝 5%를 빼도 폭 {p95 - p05:.2f}℃가 한 칸에 숨어 있다",
             transform=ax3.transAxes, ha="center", va="top", fontsize=9.5,
             color="#0b3d91")
    print(f"     확대 칸 폭: P95−P05 {p95-p05:.2f}℃ "
          f"(max−min {z.t_hat.max()-z.t_hat.min():.2f}℃ — 인용하지 않는다)")

    fig.suptitle(f"해상도 대비도 (5km vs 25m) — {day} {hh:%H시} · 5km 1칸 = 25m 40,000칸",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, f"fig5_resolution_{day}")


# ──────────────────────────────────────────────── 그림 1 (골목 확대)

# 검증 1(`core/metrics/shadow_probe.py --hour 16`)이 고른 창.
# 관악구 (483,1358) — 그늘 32% · Δt −0.556℃ · 창 안 기온폭 2.21℃ · **4피처 R² 0.82**.
# 기온폭만 보면 성북구 (503,1393)이 2.43℃로 최대지만 **R² 0.20**이다 — 그 변동을
# 그림자·건물이 설명하지 못한다. 그림의 주장이 정확히 그 설명이므로 쓰면 안 된다.
STREET_XY = (193402.0, 543399.0)     # EPSG:5186
STREET_HALF = 300.0                  # 600m × 600m


def _sweep(poly, dx: float, dy: float):
    from shapely.affinity import translate
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    parts = [poly, translate(poly, dx, dy)]
    xs, ys = poly.exterior.coords.xy
    for i in range(len(xs) - 1):
        parts.append(Polygon([(xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                              (xs[i + 1] + dx, ys[i + 1] + dy), (xs[i] + dx, ys[i] + dy)]))
    return unary_union(parts)


def _scalebar(ax, x0, y0, length=100.0, label="100 m"):
    ax.plot([x0, x0 + length], [y0, y0], color="#111", lw=2.5, solid_capstyle="butt", zorder=9)
    # 어두운 건물 위에 놓이면 글씨가 묻힌다 — 흰 바탕을 깔아준다
    ax.text(x0 + length / 2, y0 + 14, label, ha="center", va="bottom",
            fontsize=8.5, color="#111", zorder=9,
            bbox=dict(facecolor="white", alpha=0.8, pad=1.0, edgecolor="none"))


def fig1(day: str = "2026-08-06") -> None:
    import pvlib
    from shapely import from_wkt

    hh = _pick_hour(day, MAP_HOUR)
    sp = pvlib.solarposition.spa_python(
        pd.DatetimeIndex([hh]).tz_localize("Asia/Seoul"), 37.5665, 126.9780)
    el = float(sp["apparent_elevation"].iloc[0])
    az = float(sp["azimuth"].iloc[0])
    ai = int(az / AZ_STEP) % N_AZ
    # 그림자는 태양 반대 방향으로 뻗는다. 방위는 북 기준 시계방향이므로 x=sin, y=cos.
    sdir = np.deg2rad(az + 180.0)
    ux, uy = np.sin(sdir), np.cos(sdir)
    print(f"     {hh:%m-%d %H:%M} · 태양고도 {el:.1f}° · 방위 {az:.1f}° · hz_{ai:03d}")

    cx0, cy0 = STREET_XY
    H = STREET_HALF
    box = (cx0 - H, cx0 + H, cy0 - H, cy0 + H)

    # ── 격자 (창 안) ──
    g = pd.read_parquet(GRID, columns=["gx", "gy", "x", "y", "svf", "in_building",
                                       f"hz_{ai:03d}"])
    g = g[(g.x.between(box[0], box[1])) & (g.y.between(box[2], box[3]))].copy()
    g["margin"] = el - g[f"hz_{ai:03d}"]
    g["shade"] = g["margin"] < 0

    import pyarrow.parquet as pq
    p = pq.read_table(PRED[day], columns=["gx", "gy", "t_hat"],
                      filters=[("ts", "=", hh)]).to_pandas()
    g = g.merge(p, on=["gx", "gy"], how="left")
    open_cells = g[~g.in_building.astype(bool)]
    n_sh = int(open_cells.shade.sum())
    dt = (open_cells[open_cells.shade].t_hat.mean()
          - open_cells[~open_cells.shade].t_hat.mean())
    print(f"     창 안 {len(g):,}셀 (건물 밖 {len(open_cells):,}) · "
          f"그늘 {n_sh:,} ({n_sh/len(open_cells)*100:.0f}%) · "
          f"Δt {dt:+.3f}℃ · 기온 {g.t_hat.min():.2f}~{g.t_hat.max():.2f}℃")

    # ── 건물 (그림자가 창 안으로 들어올 수 있는 범위까지) ──
    MARGIN = 400.0
    b = pd.read_parquet(GRID.parent / "buildings.parquet",
                        columns=["cx", "cy", "height_m", "wkt", "geom_ok"])
    b = b[(b.cx.between(box[0] - MARGIN, box[1] + MARGIN))
          & (b.cy.between(box[2] - MARGIN, box[3] + MARGIN))
          & b.geom_ok & b.height_m.notna()].copy()
    print(f"     건물 {len(b):,}동 (창 + 여유 {MARGIN:.0f}m) · "
          f"높이 중위 {b.height_m.median():.1f}m · 최고 {b.height_m.max():.1f}m")

    geoms = [from_wkt(w) for w in b.wkt]
    L = b.height_m.to_numpy() / np.tan(np.deg2rad(el))      # 그림자 길이
    shadows = [_sweep(geo, ux * l, uy * l) for geo, l in zip(geoms, L)]

    # 25m 격자가 **얼마나 잃는가**를 잰다 — 이 그림에서 가장 정직한 숫자다.
    # ①의 그림자는 면적이고 ②는 25m 셀 판정이다. 셀보다 좁은 골목 그림자는
    # 중심점 판정에서 통째로 사라진다 (`plan.md` §한계 ③-b).
    from shapely.geometry import box as shp_box
    from shapely.ops import unary_union
    B = shp_box(box[0], box[2], box[1], box[3])
    bld_u = unary_union(geoms).intersection(B)
    sh_u = unary_union(shadows).intersection(B).difference(bld_u)
    open_ar = B.area - bld_u.area
    frac_poly = sh_u.area / open_ar
    frac_cell = n_sh / len(open_cells)
    print(f"     그늘 면적비 — 기하 {frac_poly*100:.1f}% vs 25m 격자 {frac_cell*100:.1f}% "
          f"→ **격자가 {(1-frac_cell/frac_poly)*100:.0f}%**")

    # ── 렌더 ──
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5))
    for ax in axes:
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_aspect("equal"); ax.set_facecolor("#ffffff")

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPoly

    def patches(polys, values=None, **kw):
        out, vals = [], []
        for i, geo in enumerate(polys):
            for q in (geo.geoms if geo.geom_type == "MultiPolygon" else [geo]):
                out.append(MplPoly(np.asarray(q.exterior.coords), closed=True))
                if values is not None:
                    vals.append(values[i])
        # `vmin`/`vmax`는 PatchCollection 생성자가 받지 않는다 — 만든 뒤 `set_clim`으로 준다
        clim = (kw.pop("vmin", None), kw.pop("vmax", None))
        pc = PatchCollection(out, **kw)
        if values is not None:
            pc.set_array(np.asarray(vals, float))
            if clim != (None, None):
                pc.set_clim(*clim)
        return pc

    # ① 건물 + 계산된 그림자
    #
    # 건물을 **높이로 음영**한다 (8/25). 3D 렌더 대신 택한 값싼 대안이다 —
    # 평면을 유지하므로 "열린 땅의 44.6%가 그늘"이라는 면적 수치가 안 깨지면서
    # *"우리는 건물 높이를 쓴다"*가 보인다. 그림자 길이 = 높이 ÷ tan(고도)이므로
    # **짙은 건물의 그림자가 길다**는 대응이 한 화면에서 읽힌다.
    # 근거는 `plan.md` 「3D 렌더는 1차 신청서에 하지 않는다」.
    axes[0].add_collection(patches(shadows, facecolor="#4a5b80", alpha=0.40,
                                   edgecolor="none", zorder=2))
    hgt = np.clip(b.height_m.to_numpy(float), 2.0, None)
    # 높이 분포가 극단으로 치우쳐 있다 — P50 9.6m · P95 22.3m · P99 42.2m · 최대 163m.
    # **선형 스케일로는 어느 쪽도 안 된다**: 최대값에 맞추면 저층 95%가 한 색으로
    # 뭉개지고, P98에서 자르면 163m 타워가 30m처럼 보인다. 로그로 잡는다.
    pc = patches(geoms, values=hgt, cmap="cividis_r", edgecolor="none", zorder=3,
                 norm=matplotlib.colors.LogNorm(vmin=4.0, vmax=float(hgt.max())))
    axes[0].add_collection(pc)
    cax = axes[0].inset_axes([0.58, 0.048, 0.38, 0.022])
    cb0 = fig.colorbar(pc, cax=cax, orientation="horizontal",
                       ticks=[5, 10, 20, 50, 100])
    cb0.ax.set_xticklabels(["5", "10", "20", "50", "100"])
    cb0.ax.tick_params(labelsize=7.5, pad=1)
    cb0.set_label("건물 높이 (m, 로그)", fontsize=7.5, labelpad=2)
    axes[0].set_title(f"① 실제 그림자 — 열린 땅의 {frac_poly*100:.0f}%", fontsize=10.5, pad=6)
    _scalebar(axes[0], box[0] + 40, box[2] + 40)

    # 셀을 점으로 찍으면(scatter) 25m 격자가 성기게 흩어져 지도로 안 읽힌다.
    # 격자를 2D 배열로 옮겨 `imshow`로 **셀 크기 그대로** 채운다. 건물 내부는 NaN이라
    # 흰색으로 비고, 그게 "사람이 걷는 곳만 그렸다"는 뜻이 된다.
    gx0, gy0 = int(g.gx.min()), int(g.gy.min())
    nx, ny = int(g.gx.max()) - gx0 + 1, int(g.gy.max()) - gy0 + 1
    ext = [gx0 * 25.0, (gx0 + nx) * 25.0, gy0 * 25.0, (gy0 + ny) * 25.0]

    def grid_of(sub, col):
        a = np.full((ny, nx), np.nan, np.float32)
        a[sub.gy.to_numpy() - gy0, sub.gx.to_numpy() - gx0] = sub[col].to_numpy(np.float32)
        return a

    a_shade = grid_of(open_cells.assign(v=open_cells.shade.astype(float)), "v")
    a_t = grid_of(open_cells, "t_hat")

    # ② 25m 격자가 담아낸 그늘 (모델 입력)
    axes[1].add_collection(patches(geoms, facecolor="#e8e8e8", edgecolor="#c0c0c0",
                                   lw=0.3, zorder=1))
    axes[1].imshow(np.where(a_shade > 0.5, 1.0, np.nan), extent=ext, origin="lower",
                   cmap=matplotlib.colors.ListedColormap(["#4a5b80"]),
                   vmin=0, vmax=1, interpolation="nearest", zorder=2)
    axes[1].set_title(f"② 25m 격자가 담는 그늘 — {frac_cell*100:.0f}%", fontsize=10.5, pad=6)
    _scalebar(axes[1], box[0] + 40, box[2] + 40)

    # ③ 우리 추정 기온 + **그늘 경계선**
    o = open_cells.dropna(subset=["t_hat"])
    vmin, vmax = np.nanpercentile(o.t_hat, 1), np.nanpercentile(o.t_hat, 99)
    im = axes[2].imshow(a_t, extent=ext, origin="lower", cmap="YlOrRd",
                        vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2)
    # 그늘 경계를 겹치는 것이 이 그림의 핵심이다 — "경계에서 색이 바뀐다"가 보이면
    # 위성 LST가 원리적으로 못 보는 것을 말로 설명할 필요가 없어진다.
    axes[2].contour(np.nan_to_num(a_shade), levels=[0.5], colors="#0b3d91",
                    linewidths=1.0, extent=ext, origin="lower", zorder=4)
    axes[2].add_collection(patches(geoms, facecolor="none", edgecolor="#6a6a6a",
                                   lw=0.3, zorder=3))
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cb.ax.set_title("기온 (℃)", fontsize=8.5, pad=6)
    cb.ax.tick_params(labelsize=8)
    axes[2].set_title(f"③ 우리 추정 25m",
                      fontsize=10.5, pad=6)
    _scalebar(axes[2], box[0] + 40, box[2] + 40)

    # 제목은 **그림이 실제로 보여주는 것**만 말한다. "위성이 볼 수 없는 것"은
    # 이 그림이 증명하지 않는다(위성 영상이 없다) — 그 주장은 본문에서 한다.
    fig.suptitle(f"골목 단위 3단 비교도 — 관악구 {int(2*H)}m · {day} {hh:%H시} · "
                 f"25m 격자 그림자는 {(1-frac_cell/frac_poly)*100:.0f}%다.",
                 fontsize=12.5, fontweight="bold", y=1.035)
    # 건물 수를 문자열에 박아두지 않는다 — 창이나 날짜가 바뀌면 캡션만 옛값으로 남는다
    fig.text(0.5, 0.975, f"건물 {len(b):,}동은 브이월드 실측 · 그림자는 결정론적 계산 "
                         f"(①은 건물 기하, ②는 모델이 쓰는 수평선 판정 — 지형 차폐 포함)",
             ha="center", va="top", fontsize=8.8, color="#555")
    fig.subplots_adjust(wspace=0.06)
    save(fig, f"fig1_street_{day}")


# ──────────────────────────────────────────────── 그림 6 (팬차트)

FAN = Path("data/metric/fan_cells.parquet")
DISTF = Path("data/metric/traj_dist.npz")


def fig6(day: str = "2026-08-06") -> None:
    import matplotlib.patches as mpatches
    d = pd.read_parquet(FAN)
    z = np.load(DISTF)
    R, nu = z["R"], float(z["nu"])
    L = np.linalg.cholesky(R * (nu - 2) / nu)
    rng = np.random.default_rng(20260826)

    tags = list(dict.fromkeys(d.tag))
    fig, axes = plt.subplots(1, len(tags), figsize=(5.0 * len(tags), 4.5), sharey=True)
    # y축을 90% 띠 기준으로 잡는다. **자동 축을 쓰면 t(ν=4)의 극단 표본 한둘이
    # 화면을 지배해** 정작 봐야 할 띠가 납작해진다. 극단 궤적은 그린 채로 축 밖으로
    # 나가며, 그 꼬리의 크기는 캡션에 숫자로 적는다 (아래 `TAIL_NOTE`).
    ylo, yhi = [], []
    for tag in tags:
        c = d[d.tag == tag].sort_values("ts")
        ylo.append((c.t_hat - 2.2 * c.sigma_cal).min())
        yhi.append((c.t_hat + 2.2 * c.sigma_cal).max())
    for ax, tag in zip(np.atleast_1d(axes), tags):
        c = d[d.tag == tag].sort_values("ts")
        h = c.ts.dt.hour.to_numpy()
        mu, sg = c.t_hat.to_numpy(), c.sigma_cal.to_numpy()

        g = rng.standard_normal((30, 24)) @ L.T
        w = rng.chisquare(nu, size=(30, 1)) / nu       # 궤적마다 하나 — 분산혼합
        draws = mu + sg * (g / np.sqrt(w))
        lo5, lo25, med, hi75, hi95 = np.percentile(draws, [5, 25, 50, 75, 95], axis=0)

        ax.fill_between(h, lo5, hi95, color="#c94a3f", alpha=0.16, lw=0, label="90% 구간")
        ax.fill_between(h, lo25, hi75, color="#c94a3f", alpha=0.30, lw=0, label="50% 구간")
        for k in range(30):
            ax.plot(h, draws[k], color="#7a2f28", lw=0.5, alpha=0.35, zorder=2)
        ax.plot(h, med, color="#111", lw=2.0, zorder=4, label="중위")
        ax.axhline(HOT_C, color="#0b3d91", ls="--", lw=1.2, zorder=3)
        ax.text(0.4, HOT_C + 0.15, "33℃", color="#0b3d91", fontsize=8.5, va="bottom")

        over = (draws >= HOT_C).sum(1)
        row = c.iloc[0]
        ax.set_title(f"{tag} ({row.gu})", fontsize=10.5, pad=6)
        ax.set_xlabel("시각 (KST)")
        ax.set_xticks(range(0, 24, 4)); ax.set_xlim(0, 23)
        ax.grid(alpha=0.25)
        ax.text(0.02, 0.03,
                f"SVF {row.svf:.2f} · 최근접 건물 {row.bld_d_min:.0f} m\n"
                f"33℃ 초과 {over.mean():.1f}h  (80% 구간 "
                f"{np.percentile(over,10):.0f}~{np.percentile(over,90):.0f}h)",
                transform=ax.transAxes, fontsize=8.5, va="bottom", color="#333",
                bbox=dict(facecolor="white", alpha=0.85, pad=2.5, edgecolor="none"))
    np.atleast_1d(axes)[0].set_ylabel("기온 (℃)")
    np.atleast_1d(axes)[0].set_ylim(min(ylo) - 0.5, max(yhi) + 0.5)
    np.atleast_1d(axes)[-1].legend(loc="upper right", fontsize=8.5)
    # 꼬리의 정직한 표기 — 90%·95% 구간은 conformal로 보정됐지만 **먼 꼬리는 아니다**
    # 이모지는 한글 폰트에 글리프가 없어 □로 찍힌다 (실측). 그림 안에서는 `※`를 쓴다.
    fig.text(0.5, -0.02,
             "※ 90%·95% 구간은 conformal로 보정됨. 다만 t(ν=4)의 먼 꼬리는 "
             "관측 최대(45℃)를 넘는 표본을 0.03~0.06% 생성한다 — 축 밖으로 나간 궤적이 그것이다.",
             ha="center", va="top", fontsize=8.5, color="#666")
    fig.suptitle(f"팬차트 (확률분포 시계열도) — 골목 한 곳의 하루 · 궤적 30개 · {day}",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.subplots_adjust(wspace=0.06)
    save(fig, f"fig6_fan_{day}")
    for tag in tags:
        c = d[d.tag == tag]
        print(f"     {tag:<16} 척도 {c.sigma_cal.min():.2f}~{c.sigma_cal.max():.2f}℃ · "
              f"t_hat {c.t_hat.min():.1f}~{c.t_hat.max():.1f}℃")


# ──────────────────────────────────────────────── 그림 7 (처방 지도)

TRAJ = {d: Path(f"data/metric/grid_traj_{d}.parquet")
        for d in ("2026-08-06", "2026-07-15")}
POP = Path("data/processing/grid_population.parquet")


def fig7(day: str = "2026-08-06") -> None:
    BLK = 20        # 20셀 × 25m = 500m
    import pyarrow.parquet as pq
    r = Raster()

    # ① 개입 효과 — 달력일 24시각 평균
    t = pq.read_table(PRED[day], columns=["gx", "gy", "ts", "cf_park"]).to_pandas()
    t = t[t.ts.dt.normalize() == pd.Timestamp(day)]
    eff = t.groupby(["gx", "gy"], sort=False)["cf_park"].mean().reset_index()
    del t

    # ② 노출 — 고령인구 × 33℃ 초과 시간 (사람·시간)
    ex = pd.read_parquet(TRAJ[day], columns=["gx", "gy", "hot_mean"])
    pop = pd.read_parquet(POP, columns=["gx", "gy", "pop", "pop_elder"])
    c = eff.merge(ex, on=["gx", "gy"], how="inner").merge(
        pop, on=["gx", "gy"], how="left").dropna(subset=["cf_park", "hot_mean"])
    c[["pop", "pop_elder"]] = c[["pop", "pop_elder"]].fillna(0.0)
    c["exposure"] = c.pop_elder * c.hot_mean
    print(f"     인구 — 총 {c['pop'].sum():,.0f}명 · 고령 {c.pop_elder.sum():,.0f}명 "
          f"({c.pop_elder.sum()/c['pop'].sum()*100:.1f}%) · "
          f"고령 노출 합계 **{c.exposure.sum()/1e6:.1f}백만 사람·시간**")
    print(f"     셀 {len(c):,} · 셀 단위 부호 역전 **{(c.cf_park > 0).mean()*100:.1f}%** "
          f"→ 500m 블록으로 묶는다")

    # 500m 블록 집계 — 셀 단위 GBM 계단 잡음을 상쇄시킨다
    c["bx"], c["by"] = c.gx // BLK, c.gy // BLK
    blk = c.groupby(["bx", "by"]).agg(cf_park=("cf_park", "mean"),
                                      hot_mean=("hot_mean", "mean"),
                                      exposure=("exposure", "sum"),
                                      pop_elder=("pop_elder", "sum"),
                                      n=("cf_park", "size"))
    blk = blk[blk.n >= BLK * BLK * 0.5].reset_index()      # 반 이상 채워진 블록만
    print(f"     500m 블록 {len(blk):,} · 부호 역전 **{(blk.cf_park > 0).mean()*100:.1f}%** · "
          f"효과 중위 {blk.cf_park.median():+.3f}℃ (P05 {blk.cf_park.quantile(.05):+.3f}) · "
          f"노출 중위 {blk.hot_mean.median():.1f}h")

    # 우선순위 — 둘 다 분위 순위로 바꿔 곱한다.
    # 원값을 곱하면 단위가 다른 두 양이 섞여 한쪽이 지배한다. 순위로 맞춘다.
    blk["prio"] = (-blk.cf_park).rank(pct=True) * blk.exposure.rank(pct=True)
    top = blk.nlargest(max(len(blk) // 10, 1), "prio")
    print(f"     상위 10% 블록 {len(top):,}개 · 효과 중위 {top.cf_park.median():+.3f}℃ · "
          f"고령 {top.pop_elder.sum():,.0f}명 "
          f"(**서울 고령의 {top.pop_elder.sum()/blk.pop_elder.sum()*100:.1f}%**) · "
          f"효과는 전체 중위의 {top.cf_park.median()/blk.cf_park.median():.1f}배")

    # 블록 값을 소속 셀에 되돌려 같은 래스터로 그린다
    c = c.merge(blk[["bx", "by", "cf_park", "exposure", "prio"]],
                on=["bx", "by"], how="inner", suffixes=("_cell", ""))
    gx, gy = c.gx.to_numpy(), c.gy.to_numpy()
    a_eff = r.to_array(gx, gy, c.cf_park.to_numpy())
    a_exp = r.to_array(gx, gy, c.exposure.to_numpy() / 1000.0)   # 천 사람·시간
    a_pri = r.to_array(gx, gy, c.prio.to_numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6))
    _panel(axes[0], a_eff, r, "① 소공원 개입 효과 (24시각 평균)", "YlGnBu_r", "℃",
           vmin=np.nanpercentile(a_eff, 1), vmax=np.nanpercentile(a_eff, 99))
    # 노출은 P02~P98로 잡는다. 0부터 그리면 대비가 죽는다
    _panel(axes[1], a_exp, r, "② 노출 — 고령인구 × 33℃ 초과 시간", "YlOrRd",
           "천 사람·시간", vmin=np.nanpercentile(a_exp, 2), vmax=np.nanpercentile(a_exp, 98))
    _panel(axes[2], a_pri, r, "③ 우선순위 = 효과 × 노출 (분위 순위 곱)", "magma_r", "순위 곱",
           vmin=0, vmax=1)
    # matplotlib은 마크다운을 모른다 — `**`가 문자로 찍힌다 (8/24에 이미 겪은 함정)
    fig.suptitle(f"개입 처방 지도 — 소공원 시나리오 · {day} · 500m 블록",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, 0.03,
             "※ 500m로 묶은 이유 — 25m 셀에서는 30.4%가 「개입하면 더워진다」로 뒤집힌다 "
             "(GBM 계단 잡음). 500m에서 10.8%로 줄어든다.",
             ha="center", va="top", fontsize=9, color="#666")
    fig.text(0.5, -0.005,
             "※ 개입 반사실은 인과 추론이 아니라 공간 연관 기반 추정이다. "
             "인구는 SGIS 격자통계(100m) × 집계구 성연령별인구 (이용허락범위 제한 없음).",
             ha="center", va="top", fontsize=9, color="#666")
    fig.subplots_adjust(wspace=0.02)
    save(fig, f"fig7_prescription_{day}")


# ──────────────────────────────────────────────── 그림 8 (폭염일 vs 평온일)

def fig8(hot_day: str = "2026-08-06", mild_day: str = "2026-07-15") -> None:
    import pyarrow.parquet as pq
    r = Raster()
    arr, stat = {}, {}
    for day in (hot_day, mild_day):
        hh = _pick_hour(day, MAP_HOUR)
        s = pq.read_table(PRED[day], columns=["gx", "gy", "t_hat"],
                          filters=[("ts", "=", hh)]).to_pandas()
        v = s.t_hat.to_numpy()
        arr[day] = r.to_array(s.gx.to_numpy(), s.gy.to_numpy(), v)
        stat[day] = {"med": np.median(v), "p05": np.percentile(v, 5),
                     "p95": np.percentile(v, 95), "hh": hh, "v": v}
        print(f"     {day} {hh:%H시} · 중위 {stat[day]['med']:.2f}℃ · "
              f"P95−P05 **{stat[day]['p95']-stat[day]['p05']:.2f}℃**")
    ratio = ((stat[hot_day]["p95"] - stat[hot_day]["p05"])
             / (stat[mild_day]["p95"] - stat[mild_day]["p05"]))
    print(f"     공간 편차 비 **{ratio:.2f}배**")

    # 지도마다 자기 컬러바를 다는데 `wspace`가 좁으면 **컬러바가 옆 패널 축을 덮는다**
    # (실측: 평온일 컬러바가 ③의 y축 라벨 위에 얹혔다). 넉넉히 벌린다.
    fig = plt.figure(figsize=(16.4, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.78], wspace=0.30)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    for ax, day, lab in ((ax1, hot_day, "폭염일"), (ax2, mild_day, "평온일")):
        st = stat[day]
        # 컬러바를 공통으로 두면 안 된다 — 절대 기온이 8℃ 넘게 차이 나서
        # 평온일 지도가 통째로 한 색이 된다. 날마다 자기 범위를 쓰고 폭을 비교한다.
        _panel(ax, arr[day], r, f"{lab} {day} {st['hh']:%H시} · 중위 {st['med']:.1f}℃",
               "YlOrRd", "기온 (℃)",
               vmin=np.nanpercentile(arr[day], 1), vmax=np.nanpercentile(arr[day], 99))

    # ③ 센서 관측과 격자 예측을 나란히 — 모델의 수축을 감추지 않는다
    obs = pd.read_parquet(TRAIN, columns=["ts", "tp", "sensor_ok"])
    obs = obs[obs.sensor_ok & obs.tp.notna()]
    sens = {}
    for day in (hot_day, mild_day):
        h = stat[day]["hh"]
        v = obs.loc[obs.ts == h, "tp"].to_numpy()
        sens[day] = np.percentile(v, 95) - np.percentile(v, 5)
    r_obs = sens[hot_day] / sens[mild_day]
    print(f"     센서 관측 P95−P05 폭염 {sens[hot_day]:.2f}℃ vs 평온 {sens[mild_day]:.2f}℃ "
          f"→ **{r_obs:.2f}배** · 격자는 {ratio:.2f}배 "
          f"(**모델이 폭염일 격차를 {(1-(stat[hot_day]['p95']-stat[hot_day]['p05'])/sens[hot_day])*100:.0f}% 축소**)")

    x = np.arange(2)
    ax3.bar(x - 0.19, [sens[hot_day], sens[mild_day]], 0.36,
            color="#c0392b", label="센서 관측")
    ax3.bar(x + 0.19, [stat[hot_day]["p95"] - stat[hot_day]["p05"],
                       stat[mild_day]["p95"] - stat[mild_day]["p05"]], 0.36,
            color="#e8a598", label="25m 격자 예측")
    for xi, (a, b) in enumerate([(sens[hot_day], stat[hot_day]["p95"] - stat[hot_day]["p05"]),
                                 (sens[mild_day], stat[mild_day]["p95"] - stat[mild_day]["p05"])]):
        ax3.annotate(f"{a:.2f}", (xi - 0.19, a), ha="center", va="bottom", fontsize=9)
        ax3.annotate(f"{b:.2f}", (xi + 0.19, b), ha="center", va="bottom", fontsize=9)
    ax3.set(xticks=x, xticklabels=["폭염일", "평온일"], ylabel="P95 - P05 (℃)",
            title="③ 같은 시각 셀·센서 간 격차", ylim=(0, max(sens.values()) * 1.35))
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(alpha=0.25, axis="y")
    ax3.text(0.03, 0.90, f"관측 {r_obs:.2f}배  ·  예측 {ratio:.2f}배",
             transform=ax3.transAxes, fontsize=11, fontweight="bold", color="#c0392b")
    ax3.text(0.03, 0.81, "모델은 격차를 보수적으로 낸다",
             transform=ax3.transAxes, fontsize=9, color="#666")

    fig.suptitle(f"폭염일·평온일 대비도 — 같은 {MAP_HOUR}시",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig8_hot_vs_mild")


# ──────────────────────────────────────────────── 그림 9 (다음 센서 어디에)

SITING = Path("data/metric/siting.parquet")


def fig9() -> None:
    r = Raster()
    d = pd.read_parquet(SITING)
    print(f"     셀 {len(d):,} · 피처거리 중위 {d.fdist.median():.2f} · "
          f"인구 {d['pop'].sum():,.0f}명")

    d["bx"], d["by"] = d.gx // 20, d.gy // 20
    blk = d.groupby(["bx", "by"]).agg(fdist=("fdist", "median"), pop=("pop", "sum"),
                                      n=("gx", "size"))
    blk = blk[blk.n >= 200].reset_index()
    blk["prio"] = blk.fdist.rank(pct=True) * blk["pop"].rank(pct=True)
    d = d.merge(blk[["bx", "by", "prio"]], on=["bx", "by"], how="inner")

    gx, gy = d.gx.to_numpy(), d.gy.to_numpy()
    a_f = r.to_array(gx, gy, d.fdist.to_numpy())
    a_p = r.to_array(gx, gy, d["pop"].to_numpy() * 16.0)      # 100m 환산 (셀당 → 100m당)
    a_s = r.to_array(gx, gy, d.prio.to_numpy())

    # 상위 20 블록 중심 — 지도에 점으로 찍는다
    top = blk.nlargest(20, "prio")
    tx = (top.bx.to_numpy() * 20 + 10) * 25.0 / 1000
    ty = (top.by.to_numpy() * 20 + 10) * 25.0 / 1000

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6))
    _panel(axes[0], a_f, r, "① 외삽 정도 — 비슷한 센서까지의 거리", "viridis_r", "표준편차",
           vmin=np.nanpercentile(a_f, 2), vmax=np.nanpercentile(a_f, 98))
    _panel(axes[1], a_p, r, "② 인구 (100m당)", "YlOrRd", "명",
           vmin=0, vmax=np.nanpercentile(a_p, 98))
    _panel(axes[2], a_s, r, "③ 다음 센서 우선순위 = 외삽 × 인구", "magma_r", "순위 곱",
           vmin=0, vmax=1)
    axes[2].scatter(tx, ty, s=26, facecolor="none", edgecolor="#0b3d91", lw=1.3, zorder=7)
    axes[2].scatter(tx, ty, s=3, color="#0b3d91", zorder=8)

    fig.suptitle("다음 센서 우선순위 지도 — 외삽 정도 × 인구",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, 0.03,
             "※ 파란 점 = 상위 20개 500m 블록. 외삽은 정적 피처 22개를 센서 분포로 "
             "표준화한 뒤 잰 최근접 거리다 (센서끼리의 P95 = 3.43).",
             ha="center", va="top", fontsize=9, color="#666")
    fig.text(0.5, -0.005,
             "※ 최적 설계가 아니다 — 새 센서가 불확실성을 얼마나 줄이는지가 아니라 "
             "어디가 비어 있는지를 보인다.",
             ha="center", va="top", fontsize=9, color="#666")
    fig.subplots_adjust(wspace=0.02)
    save(fig, "fig9_siting")




# ──────────────────────────────────────────────── 그림 11 (거리 감쇠)

BUFFER_CV = Path("data/metric/buffer_cv.parquet")

# 비교 기준선 (`plan.md` §3 확정값)
MAE_B0B, MAE_B1 = 0.662, 0.622          # 기상청 5km 대표값 · IDW 공간보간 단독
GRID_D_MED, GRID_D_P95 = 374.0, 1320.0  # 격자 셀의 최근접 센서 거리


def fig11() -> None:
    d = pd.read_parquet(BUFFER_CV)
    bu = d[d.kind == "버퍼"].groupby("r").agg(
        x=("d_med", "mean"), m=("mae", "mean"), s=("mae", "std")).sort_index()
    rd = d[d.kind == "무작위"].groupby("r").agg(
        x=("d_med", "mean"), m=("mae", "mean"), s=("mae", "std")).sort_index()
    base = float(bu.m.iloc[0])
    print(f"     기준선(r=0) {base:.4f}℃ · 실거리 {bu.x.iloc[0]:.0f}m")
    for r, row in bu.iterrows():
        print(f"     r={r:>4.0f} 실거리 {row.x:>5.0f}m  MAE {row.m:.4f}  "
              f"({(row.m/base-1)*100:+.1f}%)")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8),
                                  gridspec_kw={"width_ratios": [1.5, 1]})

    # ① MAE vs 실현 거리
    # y축을 0부터 잡지 않는다 — MAE가 0.55~0.65 구간이라 0을 넣으면 감쇠가 납작해진다
    ylo, yhi = 0.535, 0.685
    ax.set_ylim(ylo, yhi)
    ax.axhspan(ylo, MAE_B0B, color="#e9f3e9", zorder=0)
    ax.axhline(MAE_B0B, color="#2e7d32", lw=1.4, ls="--", zorder=2)
    ax.axhline(MAE_B1, color="#6a8caf", lw=1.2, ls=":", zorder=2)
    ax.text(440, MAE_B0B + .004, "기상청 5km 대표값 0.662", fontsize=9,
            color="#2e7d32", fontweight="bold")
    ax.text(440, MAE_B1 + .004, "IDW 공간보간 단독 0.622", fontsize=9, color="#6a8caf")

    # 격자 셀이 실제로 겪는 거리
    # 격자 셀이 실제로 겪는 거리 — 이 그림을 산출물에 연결하는 축이다
    for xv, lab in ((GRID_D_MED, f"격자 중위 {GRID_D_MED:.0f}m"),
                    (GRID_D_P95, f"격자 P95 {GRID_D_P95:,.0f}m")):
        ax.axvline(xv, color="#c94a3f", lw=1.1, ls="-.", alpha=.7, zorder=2)
        ax.text(xv + 22, yhi - .004, lab, fontsize=8.5, color="#c94a3f",
                va="top", rotation=90)

    ax.errorbar(rd.x, rd.m, yerr=rd.s, marker="s", ms=5, lw=1.6, capsize=3,
                color="#9aa3b2", label="무작위 제거 (같은 수, 대조군)", zorder=4)
    ax.errorbar(bu.x, bu.m, yerr=bu.s, marker="o", ms=6.5, lw=2.2, capsize=3,
                color="#b3122b", label="버퍼 CV (국소 공백)", zorder=5)
    for i, dx, dy in ((0, 8, -15), (3, -12, 9), (4, -46, 4)):
        ax.annotate(f"{bu.m.iloc[i]:.3f}", (bu.x.iloc[i], bu.m.iloc[i]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=9,
                    fontweight="bold", color="#b3122b")

    ax.set_xlabel("test 센서 → 남은 학습 센서의 실현 최근접 거리 중위 (m)")
    ax.set_ylabel("MAE (℃)")
    ax.set_title("① 관측에서 멀어질수록 — 그래도 5km 대표값보다 낫다", loc="left")
    ax.set_xlim(250, 2000)
    ax.legend(loc="lower right", fontsize=9)

    # ② 열화의 분해
    rs = [r for r in bu.index if r > 0]
    loc_ = [bu.m[r] - rd.m[r] for r in rs]
    den = [rd.m[r] - base for r in rs]
    xs = np.arange(len(rs))
    ax2.bar(xs, den, .58, color="#9aa3b2", label="표본·밀도 (센서가 줄어서)")
    ax2.bar(xs, loc_, .58, bottom=den, color="#b3122b", label="국소 공백 (구멍이 뚫려서)")
    for i, r in enumerate(rs):
        ax2.text(i, den[i] + loc_[i] + .0025, f"+{den[i]+loc_[i]:.3f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{bu.x[r]:.0f}m" for r in rs])
    ax2.set_xlabel("실현 최근접 거리")
    ax2.set_ylabel("MAE 증가 (℃)")
    frac = [l/(l+d)*100 for l, d in zip(loc_, den)]
    ax2.set_title(f"② 열화의 {min(frac):.0f}~{max(frac):.0f}%가 국소 공백", loc="left")
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("버퍼 CV — 평가가 스스로에게 후한지 확인한다 "
                 "(test 센서 반경 안의 학습 센서를 IDW·GBM 양쪽에서 제거)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06,
             "※ 5-fold 평균 ± 표준편차. x축은 명목 반경이 아니라 조건마다 실측한 "
             "최근접 거리다 — 격자 셀 분포와 직접 대응시키기 위해서다.\n"
             "※ 무작위 제거도 밀도를 낮춰 거리를 늘린다. 두 항의 차를 "
             "「순수 거리 효과」로 부르지 않는다.",
             ha="center", va="top", fontsize=8.8, color="#666")
    fig.tight_layout()
    save(fig, "fig11_buffer_cv")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6,
        "7": fig7, "8": fig8, "9": fig9, "11": fig11}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=None, help=f"기본 전부: {' '.join(FIGS)}")
    a = ap.parse_args()
    setup()
    for k in (a.which or list(FIGS)):
        if k not in FIGS:
            print(f"그림 {k}는 아직 없다 (가능: {' '.join(FIGS)})", file=sys.stderr)
            continue
        print(f"[그림 {k}]")
        FIGS[k]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
