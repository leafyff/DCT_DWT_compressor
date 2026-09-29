@echo off
rem Запуск програми «Стиснення зображень: DCT vs DWT (Хаар)».
rem Подвійний клік відкриває графічний інтерфейс; на start.bat можна перетягнути файл зображення.
rem З аргументами працює як команда dctdwt, напр.: start.bat --photo --batch --levels "20;5;1"
rem Відносні шляхи в аргументах рахуються від поточної теки, як у звичайної команди.
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

rem Середовище створюється, якщо його немає, а залежності ставляться, якщо їх бракує.
rem Перевірка щоразу, тож перерваний перший запуск (напр. без інтернету) доробляється наступним.
if not exist "%PY%" (
    echo Створення віртуального середовища...
    python -m venv "%ROOT%\.venv" || goto :no_python
)
"%PY%" -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in ('numpy', 'PIL', 'matplotlib')))"
if errorlevel 1 (
    echo Встановлення залежностей ^(numpy, pillow, matplotlib^)...
    "%PY%" -m pip install --quiet "%ROOT%" || goto :no_packages
)

rem Пакет dctdwt імпортується з теки проєкту, тож зміни в коді діють без перевстановлення
if defined PYTHONPATH (set "PYTHONPATH=%ROOT%;%PYTHONPATH%") else set "PYTHONPATH=%ROOT%"

rem Консоль потрібна лише пакетному режиму й довідці; інтерфейс відкривається без неї
set "CONSOLE="
for %%a in (%*) do (
    if /i "%%~a"=="--batch" set "CONSOLE=1"
    if /i "%%~a"=="-h" set "CONSOLE=1"
    if /i "%%~a"=="--help" set "CONSOLE=1"
)
if defined CONSOLE (
    "%PY%" -m dctdwt %*
    pause
) else (
    start "" "%ROOT%\.venv\Scripts\pythonw.exe" -m dctdwt %*
)
exit /b 0

:no_python
echo.
echo Не знайдено Python. Встановіть Python 3.10 або новіший з python.org і запустіть файл ще раз.
pause
exit /b 1

:no_packages
echo.
echo Не вдалося встановити залежності. Перевірте підключення до інтернету і запустіть файл ще раз.
pause
exit /b 1
