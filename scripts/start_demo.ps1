param(
    [switch]$Background,
    [switch]$Reindex
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    py -3.12 -m venv .venv
    & $pythonPath -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости' }
}
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    throw 'Заполните GOOGLE_API_KEY, DEEPDUB_API_KEY и DEEPDUB_VOICE_ID в .env, затем повторите запуск'
}
# Only legacy backends of THIS Compose project on the canonical demo port.
# Retire the identified obsolete image/container only after checking its data.
# Other project backends are stopped if conflicting, never deleted here.
$legacyIds = @(docker ps -aq --filter 'label=com.docker.compose.project=deep' --filter 'label=com.docker.compose.service=voice-rag')
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop недоступен' }
foreach ($containerId in $legacyIds) {
    if (-not $containerId) { continue }
    $containerInfo = @(docker inspect $containerId | ConvertFrom-Json)[0]
    $bindings = $containerInfo.HostConfig.PortBindings.'8000/tcp'
    if (@($bindings | Where-Object { $_.HostPort -eq '8010' }).Count -eq 0) { continue }
    docker update --restart=no $containerId | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось отключить автозапуск старого backend' }
    if ($containerInfo.State.Running) {
        Write-Host 'Останавливаю старый backend проекта на порту 8010.'
        docker stop $containerId | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось остановить старый backend' }
    }
    $obsoleteImage = 'sha256:027637f6debb87d7c320ecdf860f94a8a381c93fb0b2d5fea1b4975a4e9ce983'
    if ($containerInfo.Name -eq '/constitution-voice-rag' -and $containerInfo.Image -eq $obsoleteImage) {
        $expectedDataPath = [IO.Path]::GetFullPath((Join-Path $projectRoot 'data'))
        $mounts = @($containerInfo.Mounts)
        $safeMount = $mounts.Count -eq 1 -and $mounts[0].Type -eq 'bind' -and $mounts[0].Destination -eq '/app/data' -and [IO.Path]::GetFullPath($mounts[0].Source) -eq $expectedDataPath
        $changes = @(docker diff $containerId)
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось проверить файлы старого контейнера' }
        $unexpectedChanges = @($changes | Where-Object {
            $_ -notmatch '^C /usr(?:/local(?:/lib(?:/python3\.12(?:/[A-Za-z0-9_]+)*)?)?)?$' -and
            $_ -notmatch '^A /usr/local/lib/python3\.12/(?:[A-Za-z0-9_]+/)*__pycache__/[A-Za-z0-9_.-]+\.pyc$'
        })
        if (-not $safeMount -or $unexpectedChanges.Count -gt 0) {
            throw 'В старом контейнере обнаружены непроверенные файлы/тома. Он остановлен, но сохранён для проверки.'
        }
        Write-Host 'Удаляю устаревший контейнер Конституции из Docker Desktop. Данные и образ сохраняются.'
        docker rm $containerId | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось удалить устаревший контейнер' }
    }
}

$baseUrl = 'http://127.0.0.1:8010'
function Test-DemoApi {
    try {
        $schema = Invoke-RestMethod "$baseUrl/openapi.json" -TimeoutSec 2
        if ($schema.info.title -ne 'Kaspi Gold Voice Demo' -or $null -eq $schema.paths.'/api/calls'.post) { return $false }
        $health = Invoke-RestMethod "$baseUrl/health" -TimeoutSec 2
        return $health.status -eq 'ready' -and $health.indexed_passages -gt 0 -and $health.retrieval -eq 'qdrant-hybrid'
    } catch { return $false }
}
$listener = Get-NetTCPConnection -State Listen -LocalPort 8010 -ErrorAction SilentlyContinue
if ($listener) {
    if (Test-DemoApi) {
        docker compose up -d livekit qdrant
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить LiveKit' }
        Write-Host "Демо уже запущено: $baseUrl"
        return
    }
    $ownerIds = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw "Порт 8010 занят другим приложением (PID: $ownerIds). Освободите порт и повторите запуск. Процесс не остановлен автоматически."
}
docker compose up -d livekit qdrant
if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить LiveKit. Проверьте Docker Desktop' }
if ($Reindex -or -not (Test-Path -LiteralPath 'data/active_corpus.json')) {
    & $pythonPath -m scripts.fetch_corpus
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось извлечь официальные источники' }
    & $pythonPath -m scripts.publish_corpus
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось опубликовать индекс источников' }
}
if ($Background) {
    $logDirectory = Join-Path $projectRoot 'artifacts/runtime'
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8010') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'backend.stdout.log') -RedirectStandardError (Join-Path $logDirectory 'backend.stderr.log') -PassThru
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        if ($serverProcess.HasExited) { throw 'Сервер завершился. Диагностика: artifacts/runtime/backend.stderr.log' }
        if (Test-DemoApi) {
            $serverProcess.Id | Set-Content -LiteralPath (Join-Path $logDirectory 'backend.pid')
            Write-Host "Демо запущено: $baseUrl (PID $($serverProcess.Id)). Логи: artifacts/runtime"
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw 'Сервер не ответил за 20 секунд. Диагностика: artifacts/runtime/backend.stderr.log'
}
Write-Host "Демо: $baseUrl . Ctrl+C завершает сервер приложения."
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port 8010
if ($LASTEXITCODE -ne 0) { throw 'Не удалось запустить сервер приложения' }
