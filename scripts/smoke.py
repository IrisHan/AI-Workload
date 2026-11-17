import argparse
import json
import re
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8666")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    results = []

    def get(path):
        with urlopen(base + path, timeout=10) as response:
            content = response.read()
            results.append({"path": path, "status": response.status, "bytes": len(content)})
            return content

    assert json.loads(get("/api/health"))["status"] == "ok"
    models = json.loads(get("/api/models"))
    totals = {m["repo_id"]: m["parameters"] for m in models}
    assert totals["Qwen/Qwen2.5-0.5B"] == 494032768
    assert totals["Qwen/Qwen2.5-1.5B"] == 1543714304
    assert json.loads(get("/api/hardware"))[0]["bf16_tflops"] == 989.5
    assert isinstance(json.loads(get("/api/workloads/designs"))["items"], list)
    assert json.loads(get("/api/workloads/capabilities"))["runtime_profiles"]
    paths = json.loads(get("/openapi.json"))["paths"]
    assert "/api/workloads/runs/{identifier}/plans/{pid}/inspection" in paths
    assert "/api/workloads/{identifier}/profile" not in paths
    html = get("/models").decode()
    assert '<div id="root">' in html
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
    assert len(assets) >= 2
    for asset in assets:
        assert get(asset)
    for route in ["/workloads/inference/new", "/workloads/compare", "/workloads/old", "/hardware", "/planning/resources"]:
        assert get(route) == html.encode()
    try:
        get("/api/not-a-route")
        raise AssertionError("Unknown API route must be 404")
    except HTTPError as exc:
        assert exc.code == 404
        results.append({"path": "/api/not-a-route", "status": 404})
    try:
        get("/api/workloads")
        raise AssertionError("Legacy Workload API must be 410")
    except HTTPError as exc:
        assert exc.code == 410
        assert json.loads(exc.read())["error"]["code"] == "LEGACY_WORKLOAD_REMOVED"
        results.append({"path": "/api/workloads", "status": 410})
    print(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                      "base_url": base, "checks": results, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
