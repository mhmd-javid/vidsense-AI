import os
import urllib.request
import json

# ============================================================
# ۱. تست کلید API
# ============================================================
key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    print("❌ OPENROUTER_API_KEY در محیط تنظیم نشده است!")
    print("   دستور setx OPENROUTER_API_KEY 'sk-or-...' را اجرا کنید.")
    exit(1)

print(f"✅ کلید پیدا شد: {key[:10]}... (ادامه مخفی)")

# ============================================================
# ۲. انتخاب مدل (Qwen 2.5 7B - رایگان)
# ============================================================
MODEL = "qwen/qwen-2.5-7b-instruct:free"

# ============================================================
# ۳. ارسال درخواست به OpenRouter
# ============================================================
body = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Respond in Persian."},
        {"role": "user", "content": "سلام، چطوری؟ لطفاً فقط پاسخ بده، هیچ توضیح اضافی نده."}
    ],
    "max_tokens": 50,
    "temperature": 0
}

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps(body).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "VidSense"
    },
    method="POST"
)

print(f"\n🔄 ارسال درخواست به OpenRouter با مدل: {MODEL} ...")

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        reply = result["choices"][0]["message"]["content"]
        print(f"\n✅ پاسخ OpenRouter:\n{reply}")
        print(f"\n✅ مدل {MODEL} فعال و پاسخ می‌دهد!")
except urllib.error.HTTPError as e:
    print(f"\n❌ خطای HTTP {e.code}: {e.reason}")
    body = e.read().decode("utf-8") if e.fp else ""
    if "404" in str(e):
        print("   مدل پیدا نشد. آدرس مدل را بررسی کنید.")
    elif "401" in str(e):
        print("   کلید API نامعتبر است یا منقضی شده.")
    elif "429" in str(e):
        print("   محدودیت درخواست روزانه پر شده است.")
    else:
        print(f"   جزئیات: {body[:300]}")
except Exception as e:
    print(f"\n❌ خطای دیگر: {e}")