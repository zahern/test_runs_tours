$root = "runs\ga_parallel"
$out = @()
Get-ChildItem $root -Directory | Sort-Object Name | ForEach-Object {
  $tag = $_.Name
  $stageDirs = Get-ChildItem "$($_.FullName)" -Directory -Filter "stage*" -ErrorAction SilentlyContinue
  foreach ($sd in $stageDirs) {
    $shardRoot = Join-Path $sd.FullName "shards"
    if (-not (Test-Path $shardRoot)) { continue }
    foreach ($sh in (Get-ChildItem $shardRoot -Directory -ErrorAction SilentlyContinue)) {
      $gen = 0; $genFile = $null
      $hist = Get-ChildItem $sh.FullName -Filter "ga_history_partial_*.csv" -ErrorAction SilentlyContinue
      foreach ($h in $hist) {
        $lines = @(Get-Content $h.FullName -ErrorAction SilentlyContinue | Where-Object { $_.Trim() -ne "" })
        if ($lines.Count -ge 2) {
          $last = $lines[-1].Split(',')[0]
          $v = 0
          if ([int]::TryParse($last, [ref]$v)) {
            if ($v -ge $gen) { $gen = $v; $genFile = $h.Name }
          }
        }
      }
      $imp = 0; $gain = $null; $ff = $null; $lf = $null
      $ct = Join-Path $sh.FullName "convergence_trace.csv"
      if (Test-Path $ct) {
        $cl = @(Get-Content $ct -ErrorAction SilentlyContinue | Where-Object { $_.Trim() -ne "" })
        if ($cl.Count -ge 2) {
          $vals = @()
          for ($i=1; $i -lt $cl.Count; $i++) {
            $p = $cl[$i] -split '","'
            $fv = $null
            if ($p.Count -ge 3) { [double]::TryParse($p[1], [ref]$fv) | Out-Null }
            if ($fv -ne $null) { $vals += $fv }
          }
          if ($vals.Count -gt 0) {
            $imp = $vals.Count
            $ff = [math]::Round($vals[0],4)
            $lf = [math]::Round(($vals | Measure-Object -Maximum).Maximum,4)
            $gain = [math]::Round($lf - $ff, 6)
          }
        }
      }
      $bestNpz = @(Get-ChildItem $sh.FullName -Filter "best_*.npz" -ErrorAction SilentlyContinue)
      $bestNonZero = @($bestNpz | Where-Object { $_.Length -gt 0 })
      $gaPkl = Test-Path (Join-Path $sh.FullName "ga_results.pkl")
      $gaFail = Test-Path (Join-Path $sh.FullName "ga_results.FAILED.pkl")
      $partial = @(Get-ChildItem $sh.FullName -Filter "PARTIAL*" -ErrorAction SilentlyContinue).Count -gt 0
      $failed = ""
      $pj = Join-Path $sh.FullName "ga_progress.json"
      if (Test-Path $pj) {
        try { $failed = (Get-Content $pj -Raw | ConvertFrom-Json).failed } catch { $failed = "parse_err" }
      }
      $out += [pscustomobject]@{
        Tag=$tag; Stage=$sd.Name; Shard=$sh.Name; MaxGen=$gen
        HistFiles=@($hist).Count; Imp=$imp; FirstF=$ff; BestF=$lf; Gain=$gain
        BestNpz=$bestNpz.Count; NpzOK=$bestNonZero.Count
        Results=$gaPkl; Failed=$gaFail; Partial=$partial; GAFailed=$failed
        MTime=$sh.LastWriteTime.ToString("MM-dd HH:mm")
      }
    }
  }
}
$out | Export-Csv "_conv_audit.csv" -NoTypeInformation
Write-Output ("rows=" + $out.Count)