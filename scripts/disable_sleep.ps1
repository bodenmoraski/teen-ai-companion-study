# PowerShell script to disable sleep (run as Administrator)
# Right-click → Run as Administrator

Write-Host "Disabling Windows sleep/hibernate..." -ForegroundColor Yellow

# Disable sleep when plugged in
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 30

# Disable sleep on battery (if laptop)
powercfg /change standby-timeout-dc 0

Write-Host "Done! Windows will not sleep while plugged in." -ForegroundColor Green
Write-Host "Display will turn off after 30 minutes (process continues running)" -ForegroundColor Green
Write-Host ""
Write-Host "To re-enable sleep later, run:" -ForegroundColor Yellow
Write-Host "  powercfg /change standby-timeout-ac 30" -ForegroundColor Yellow

