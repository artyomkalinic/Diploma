from pathlib import Path
import shutil

# Запускается из корня проекта, если рядом лежат CSV, сохранённые ноутбуками.
# Скрипт просто переносит/копирует predictions*.csv в ./data.

root = Path(".")
data = root / "data"
data.mkdir(exist_ok=True)

patterns = [
    "predictions_poisson_rolling_*.csv",
    "predictions_dc_rolling_*.csv",
    "predictions_dc_ext_rolling_*.csv",
    "predictions_poisson_xg_*.csv",
]

copied = 0
for pattern in patterns:
    for file in root.glob(pattern):
        target = data / file.name
        shutil.copy2(file, target)
        print(f"copied {file} -> {target}")
        copied += 1

if copied == 0:
    print("CSV-файлы predictions*.csv не найдены рядом со скриптом.")
else:
    print(f"Готово. Скопировано файлов: {copied}")
