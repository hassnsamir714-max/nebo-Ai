# Nebo AI

مشروع تعليمي صغير لتدريب نموذج لغوي حرفي من الصفر باستخدام Python وPyTorch.
هذا المشروع يشرح فكرة تعلم الأنماط في النصوص، ثم توليد نص جديد بناءً على تلك الأنماط.

## ما الذي يحتويه المشروع؟
- `model.py` — تعريف شبكة عصبية صغيرة
- `train.py` — تدريب النموذج وحفظه
- `app.py` — تشغيل النموذج لتوليد النص
- `data/sample_data.txt` — نص تجريبي للتدريب
- `requirements.txt` — الاعتمادات اللازمة

## التثبيت
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
