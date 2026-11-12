# Owner action (audit 6, A-63 / D-052): make the nightly forward-test task robust to sleep, restarts and log-off.
# Run it yourself in an ADMINISTRATOR PowerShell:   powershell -ExecutionPolicy Bypass -File scripts\shadow\owner_fix_task.ps1
# The assistant never runs this script: changing a scheduled task is the owner's decision.
#
# What it does to the task "NEXUS Alpha shadow test" (nothing else on the PC is changed):
#   1. Wake the computer to run the task (Conditions tab), run it as soon as possible after a missed start, and let it
#      start and continue on battery power (a laptop unplugged at 07:00 would otherwise skip the run).
#   2. Allow wake timers in the active power plan (on AC power and on battery); without this, step 1 has no effect.
#   3. Add two extra triggers: at log-on and on workstation unlock. A later run only works if the earlier one did not
#      complete: the done marker is keyed by the as-of session (A-64), so extra runs are harmless.
#   4. Optionally (you are asked), run whether you are logged on or not. It stores your Windows password with the
#      task, so no console window can be closed by accident and the run survives a lock screen. Answer N to skip.
$ErrorActionPreference = "Stop"
$name = "NEXUS Alpha shadow test"
$task = Get-ScheduledTask -TaskName $name
$user = "$env:USERDOMAIN\$env:USERNAME"

# 1. wake timer and catch-up
$set = $task.Settings
$set.WakeToRun = $true
$set.StartWhenAvailable = $true
$set.DisallowStartIfOnBatteries = $false
$set.StopIfGoingOnBatteries = $false

# 2. wake timers allowed in the active power plan (1 = enable)
powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
powercfg /setactive SCHEME_CURRENT

# 3. extra triggers: keep the existing ones, add log-on and unlock if missing
$triggers = @($task.Triggers)
if (-not ($triggers | Where-Object { $_.CimClass.CimClassName -eq "MSFT_TaskLogonTrigger" })) {
    $triggers += New-ScheduledTaskTrigger -AtLogOn -User $user
}
if (-not ($triggers | Where-Object { $_.CimClass.CimClassName -eq "MSFT_TaskSessionStateChangeTrigger" })) {
    $cls = Get-CimClass -ClassName MSFT_TaskSessionStateChangeTrigger -Namespace Root/Microsoft/Windows/TaskScheduler
    $unlock = New-CimInstance -CimClass $cls -ClientOnly
    $unlock.StateChange = 8          # TASK_SESSION_UNLOCK
    $unlock.UserId = $user
    $unlock.Enabled = $true
    $triggers += $unlock
}
Set-ScheduledTask -TaskName $name -Settings $set -Trigger $triggers | Out-Null
Write-Host "Wake timer on, catch-up on, log-on and unlock triggers present."

# 4. optional: run whether logged on or not (asks for your Windows password; it is stored by Windows, not by us)
$ans = Read-Host "Also run the task whether you are logged on or not? It needs your Windows password once (y/N)"
if ($ans -match "^[yY]") {
    $cred = Get-Credential -UserName $user -Message "Windows password for the scheduled task"
    Set-ScheduledTask -TaskName $name -User $cred.UserName -Password $cred.GetNetworkCredential().Password | Out-Null
    Write-Host "The task now runs whether you are logged on or not."
}

$t = Get-ScheduledTask -TaskName $name
Write-Host ("WakeToRun={0}  StartWhenAvailable={1}  LogonType={2}  Triggers={3}" -f $t.Settings.WakeToRun,
    $t.Settings.StartWhenAvailable, $t.Principal.LogonType, $t.Triggers.Count)
Write-Host "Keep the laptop plugged in overnight. Sleep is fine; shut down is not (a wake timer cannot start a powered-off PC)."
