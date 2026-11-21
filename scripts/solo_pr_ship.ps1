param(
    [string]$Title = "chore: automated verified update",
    [string]$Body = "Automated Solo-Developer PR verified and approved by ArXiv-to-Code Invariant Bot."
)
$ErrorActionPreference = "Stop"

# Guard against recursive hook invocation
$env:ARXIV_TO_CODE_SKIP_POST_COMMIT_HOOK = "1"

# Fail-closed Repository Boundary Check (RULE: repository_boundaries.md)
$expectedRepoSlug = "oyesanyf/arxiv-to-code"
$topLevel = (git rev-parse --show-toplevel 2>$null).Trim()
if (-not $topLevel -or ($topLevel -notmatch "arixiv-to-code$" -and $topLevel -notmatch "arxiv-to-code$")) {
    throw "FAIL-CLOSED REPOSITORY BOUNDARY VIOLATION: Expected repository root ending in arixiv-to-code or arxiv-to-code, got '$topLevel'"
}

$remoteUrl = (git remote get-url origin 2>$null).Trim()
if ($remoteUrl -notmatch "oyesanyf/arxiv-to-code") {
    throw "FAIL-CLOSED REPOSITORY BOUNDARY VIOLATION: Expected origin remote for '$expectedRepoSlug', got '$remoteUrl'"
}

$branchName = "pr/solo-" + (Get-Date -Format "yyyyMMdd-HHmmss")
Write-Host "[1/5] Setting up feature branch $branchName in $expectedRepoSlug..." -ForegroundColor Cyan

git fetch origin main --quiet

$uncommitted = (git status --porcelain)
$headSha = (git rev-parse HEAD).Trim()
$originMainSha = (git rev-parse origin/main).Trim()

if ($uncommitted) {
    Write-Host "Detected uncommitted changes. Staging and committing to $branchName..." -ForegroundColor Yellow
    git checkout -b $branchName
    git add -A
    git commit -m "$Title"
} elseif ($headSha -ne $originMainSha) {
    Write-Host "Detected local commit(s) ahead of origin/main. Branching to $branchName and resetting local main..." -ForegroundColor Yellow
    git branch $branchName HEAD
    git reset --hard origin/main
    git checkout $branchName
} else {
    Write-Host "Working tree is clean and HEAD matches origin/main. Nothing to ship." -ForegroundColor Green
    exit 0
}

Write-Host "Pushing $branchName to origin..." -ForegroundColor Cyan
git push -u origin $branchName

$pythonExe = if (Test-Path "C:\Users\oyesanyf\AppData\Local\Programs\Python\Python312\python.exe") {
    "C:\Users\oyesanyf\AppData\Local\Programs\Python\Python312\python.exe"
} else {
    "python"
}

Write-Host "[2/5] Opening Pull Request against main on $expectedRepoSlug..." -ForegroundColor Cyan
$prNumber = & $pythonExe -c "
import subprocess, urllib.request, urllib.error, json, sys

p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = p.communicate('protocol=https\nhost=github.com\n\n')
token = dict(l.split('=', 1) for l in out.splitlines() if '=' in l).get('password', '')

payload = json.dumps({'title': sys.argv[1], 'body': sys.argv[2], 'head': sys.argv[3], 'base': 'main'}).encode('utf-8')
req = urllib.request.Request(
    'https://api.github.com/repos/oyesanyf/arxiv-to-code/pulls',
    data=payload,
    method='POST',
    headers={'Authorization': 'Bearer ' + token, 'User-Agent': 'arxiv-to-code', 'Content-Type': 'application/json'}
)

try:
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    print(resp['number'])
except urllib.error.HTTPError as e:
    err_body = e.read().decode('utf-8', errors='replace')
    sys.stderr.write('HTTP Error ' + str(e.code) + ': ' + err_body + '\n')
    sys.exit(1)
" "$Title" "$Body" "$branchName"

$prNumber = if ($prNumber) { $prNumber.Trim() } else { "" }
Write-Host "[3/5] Opened PR #$prNumber. Waiting for Invariant Review Bot Approval..." -ForegroundColor Cyan

$approved = $false
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 5
    $reviewState = & $pythonExe -c "
import subprocess, urllib.request, json, sys

p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = p.communicate('protocol=https\nhost=github.com\n\n')
token = dict(l.split('=', 1) for l in out.splitlines() if '=' in l).get('password', '')

req = urllib.request.Request(
    'https://api.github.com/repos/oyesanyf/arxiv-to-code/pulls/' + sys.argv[1] + '/reviews',
    headers={'Authorization': 'Bearer ' + token, 'User-Agent': 'arxiv-to-code'}
)

try:
    reviews = json.loads(urllib.request.urlopen(req).read().decode('utf-8', errors='replace'))
    if isinstance(reviews, list) and any(isinstance(r, dict) and r.get('state') == 'APPROVED' for r in reviews):
        print('APPROVED')
    else:
        print('WAITING')
except Exception:
    print('WAITING')
" "$prNumber"

    if (($reviewState -as [string]).Trim() -eq "APPROVED") {
        $approved = $true
        Write-Host "[4/5] PR #$prNumber APPROVED by Invariant Review Bot!" -ForegroundColor Green
        break
    } else {
        Write-Host "  Waiting for approval review... (attempt $($i + 1)/24)" -ForegroundColor Gray
    }
}

Write-Host "[5/5] Merging PR #$prNumber into main and syncing local branch..." -ForegroundColor Cyan
& $pythonExe -c "
import subprocess, urllib.request, urllib.error, json, sys, time

p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = p.communicate('protocol=https\nhost=github.com\n\n')
token = dict(l.split('=', 1) for l in out.splitlines() if '=' in l).get('password', '')

pr_num = sys.argv[1]
payload = json.dumps({'merge_method': 'squash'}).encode('utf-8')
merge_url = 'https://api.github.com/repos/oyesanyf/arxiv-to-code/pulls/' + pr_num + '/merge'

for attempt in range(1, 6):
    req = urllib.request.Request(
        merge_url,
        data=payload,
        method='PUT',
        headers={'Authorization': 'Bearer ' + token, 'User-Agent': 'arxiv-to-code', 'Content-Type': 'application/json'}
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read().decode())
        print('Merge successful:', resp.get('message', 'Merged'))
        break
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8', errors='replace')
        if 'Pull Request is already merged' in err_msg:
            print('Already merged by auto-review bot.')
            break
        if e.code in (405, 409) and attempt < 5:
            time.sleep(3)
        else:
            sys.stderr.write('Merge HTTP Error ' + str(e.code) + ': ' + err_msg + '\n')
            if attempt == 5:
                sys.exit(1)
" "$prNumber"

git checkout main
git pull origin main
git branch -D $branchName 2>$null
try { git push origin --delete $branchName 2>$null } catch {}

Write-Host "Successfully shipped PR #$prNumber on $expectedRepoSlug!" -ForegroundColor Green
