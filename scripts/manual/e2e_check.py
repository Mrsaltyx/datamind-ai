import json
import urllib.request
import uuid

BASE = "http://localhost:8124"


def post(path, body=None, headers=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else b"{}",
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def post_csv(path, csv_path):
    boundary = uuid.uuid4().hex
    raw = open(csv_path, "rb").read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"data.csv\"\r\nContent-Type: text/csv\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


# 1. Health (mode degrade attendu)
health = json.load(urllib.request.urlopen(BASE + "/api/health"))
print("HEALTH:", health["status"], "| llm:", health["llm"], "|", health["llm_message"][:60])

# 2. Upload
up = post_csv("/api/data/upload", "tests/fixtures/sample_heart.csv")
sid = up["session_id"]
print("UPLOAD: session", sid[:8], "| shape", up["summary"]["shape"])

# 3. Train baseline (aucun LLM requis)
train = post(f"/api/ml/{sid}/train")
print("TRAIN:", train["success"], "|", train["task_type"], "|", train["model_name"])
for m in train["metrics"]:
    print(f"   {m['name']}: {m['mean']:.4f} ± {m['std']:.4f}")

# 4. Chat en mode degrade -> doit renvoyer une erreur exploitable
try:
    post(f"/api/chat/{sid}/send", {"message": "test"})
    print("CHAT: ERREUR - aurait du echouer")
except urllib.error.HTTPError as e:
    detail = json.load(e)["detail"]
    print("CHAT degrade:", e.code, "|", detail[:70])

# 5. Suggest ML (ne necessite pas de LLM)
suggest = post(f"/api/ml/{sid}/suggest")
print("SUGGEST:", suggest["success"], "|", suggest["text"].splitlines()[0][:70])
print("E2E OK")
