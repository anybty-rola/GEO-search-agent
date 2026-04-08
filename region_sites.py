from __future__ import annotations

from typing import Dict, List, Literal, Tuple

RegionKey = Literal["cn", "global", "mixed"]


def get_platform_sites(region: str) -> Dict[str, List[str]]:
    """按地区返回 DuckDuckGo site: 过滤用的站点分组（近似「国内 / 海外 / 混合」信源）。"""
    r = (region or "mixed").strip().lower()
    if r == "cn":
        return _CN.copy()
    if r == "global":
        return _GLOBAL.copy()
    return _MIXED.copy()


def get_region_label(region: str) -> str:
    r = (region or "").strip().lower()
    if r == "cn":
        return "国内站点群"
    if r == "global":
        return "海外站点群"
    return "混合（国内+海外）"


def platform_choices_for_region(region: str) -> Tuple[List[str], List[str]]:
    """
    返回 (可选平台 key 列表, 默认勾选)。
    不含「通用网页」——通用网页始终会搜。
    """
    m = get_platform_sites(region)
    keys = [k for k, sites in m.items() if k != "通用网页" and sites]
    r = (region or "mixed").strip().lower()
    if r == "cn":
        default = [k for k in ("知乎", "小红书", "电商") if k in keys]
    elif r == "global":
        default = [k for k in ("Reddit", "YouTube", "电商") if k in keys]
    else:
        default = [k for k in ("知乎", "Reddit", "电商") if k in keys]
    if not default:
        default = keys[:3]
    return keys, default


# --- 预设站点（可按需再扩充） ---
_CN: Dict[str, List[str]] = {
    "通用网页": [],
    "知乎": ["zhihu.com"],
    "小红书": ["xiaohongshu.com"],
    "B站": ["bilibili.com"],
    "电商": ["jd.com", "taobao.com", "tmall.com"],
    "数码导购": ["smzdm.com", "sspai.com"],
    "品牌官网": [],
}

_GLOBAL: Dict[str, List[str]] = {
    "通用网页": [],
    "Reddit": ["reddit.com"],
    "YouTube": ["youtube.com"],
    "电商": ["amazon.com", "bestbuy.com"],
    "评测": ["thewirecutter.com", "rtings.com"],
    "品牌官网": [],
}

_MIXED: Dict[str, List[str]] = {
    "通用网页": [],
    "Reddit": ["reddit.com"],
    "YouTube": ["youtube.com"],
    "知乎": ["zhihu.com"],
    "小红书": ["xiaohongshu.com"],
    "电商": [
        "amazon.com",
        "jd.com",
        "taobao.com",
    ],
    "评测": ["thewirecutter.com", "smzdm.com"],
    "品牌官网": [],
}
