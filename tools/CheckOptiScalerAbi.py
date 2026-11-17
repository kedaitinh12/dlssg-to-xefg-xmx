#!/usr/bin/env python3
"""Check the mod's plugin v1 declaration prefix against a pinned OptiScaler SDK.
This is a source contract check, not a Windows binary or game test.
"""
import argparse
from pathlib import Path
import re
import urllib.request

REVISION = '03b37db73979e042bce4be59f44d37f348ac6905'
URL = f'https://raw.githubusercontent.com/optiscaler/OptiScaler/{REVISION}/plugin_sdk/optiscaler_plugin.h'
ROOT = Path(__file__).resolve().parents[1]

def clean(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)

def fields(text):
    body = re.search(r'typedef\s+struct\s+OptiPluginApi\s*\{(.*?)\}\s*OptiPluginApi', clean(text), re.S)
    if not body:
        raise ValueError('OptiPluginApi declaration missing')
    return [re.sub(r'\s+', '', field) for field in body[1].split(';') if field.strip()]

def check(local, upstream):
    ours, host = fields(local), fields(upstream)
    if not ours or host[:len(ours)] != ours:
        raise ValueError('OptiScaler ABI prefix changed: review field ordering and signatures')
    for pattern in (r'typedef\s+void\s*\(\*OptiDrawTabFn\).*?;',
                    r'typedef\s+bool\s*\(\*PFN_OptiPluginInit\).*?;',
                    r'typedef\s+enum\s+OptiLogLevel\s*\{.*?\}\s*OptiLogLevel;'):
        a, b = re.search(pattern, clean(local), re.S), re.search(pattern, clean(upstream), re.S)
        if not a or not b or re.sub(r'\s+', '', a[0]) != re.sub(r'\s+', '', b[0]):
            raise ValueError('Plugin callback/log type changed')
    return len(ours)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header', type=Path, help='Use an already downloaded upstream header (offline)')
    args = parser.parse_args()
    upstream = args.header.read_text() if args.header else urllib.request.urlopen(URL, timeout=60).read().decode()
    count = check((ROOT / 'src/optiscaler_plugin.h').read_text(), upstream)
    print(f'PASS: {count} API fields match the OptiScaler SDK prefix at {REVISION}; callbacks/log enum match.')
