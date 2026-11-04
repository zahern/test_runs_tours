$stagesExpected = [ordered]@{
  "0"  = @("all"); "1" = @("work","edu","nonmand"); "2" = @("start","end","budget");
  "3"  = @("work","edu","nonmand"); "4" = @("work","edu","nonmand");
  "5"  = @("work","edu","nonmand");
  "51" = @("work","edu","escort","shopping","othmaint","othdiscr","eat","social")
}
$minGens = @{ "0"=20; "1"=20; "2"=20; "3"=20; "4"=120; "5"=20; "51"=20 }
$patience= @{ "0"=24; "1"=24; "2"=24; "3"=24; "4"=120; "5"=24; "51"=24 }

$genRe = [regex]'Gen\s+(\d+)\s*\|\s*best_BIC/n=([-\d\.]+)\s+mean_BIC/n=([-\d\.]+)\s+n_feat=(\d+)\s+div=([\d\.]+)\s+mut=([\d\.]+)\s+stag=(\d+)'
$agg = @{}
Get-ChildItem . -Filter "ga_*.o*" -File -ErrorAction SilentlyContinue | ForEach-Object {
  $name = $_.Name
  if ($name -notmatch '^ga_s(?<stage>\d+)(?<seg>[a-z]+)_(?<tag>.+)\.(o|e)\d+$') { return }
  $st=$Matches['stage']; $seg=$Matches['seg']; $tg=$Matches['tag']
  $txt = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
  if ($null -eq $txt) { return }
  $m = $genRe.Matches($txt); if ($m.Count -eq 0) { return }
  $key = "$tg|$st|$seg"
  if (-not $agg.ContainsKey($key)) {
    $agg[$key] = [pscustomobject]@{ Tag=$tg; Stage=$st; Seg=$seg
      MaxGen=0; BestBICn=1e30; FirstBICn=$null; LastStag=0; Blocks=0
      EarlyStop=$false; Bad=0; Files=0; LastFile=$name; BlocksList="" }
  }
  $a = $agg[$key]; $a.Files++
  $vals=@()
  foreach ($x in $m) {
    $g=[int]$x.Groups[1].Value
    $f=[double]$x.Groups[2].Value
    if ($f -le -1e8) { $a.Bad++ } else { $vals += ,@($g,$f,[int]$x.Groups[7].Value) }
  }
  if ($vals.Count -eq 0) { $a.LastFile=$name; continue }
  $a.Blocks += ($m | ForEach-Object { if ([int]$_.Groups[1].Value -eq 0) {1} else {0} } | Measure-Object -Sum).Sum
  foreach ($v in $vals) {
    if ($v[0] -gt $a.MaxGen) { $a.MaxGen = $v[0] }
    if ($v[1] -lt $a.BestBICn) { $a.BestBICn = $v[1] }
  }
  $a.FirstBICn = $vals[0][1]
  $a.LastStag  = $vals[-1][2]
  if ($txt -match 'Early stop') { $a.EarlyStop = $true }
  if ($name -like "*.o*") { $a.LastFile = $name }
}
$rows = @()
foreach ($k in $agg.Keys) {
  $a = $agg[$k]
  $mg = $minGens[$a.Stage]; $pt = $patience[$a.Stage]
  $verdict =
    if ($a.EarlyStop -and $a.MaxGen -ge $mg) { "CONVERGED" }
    elseif ($a.MaxGen -ge $mg -and $a.LastStag -ge $pt) { "CONVERGED(noES)" }
    elseif ($a.MaxGen -ge $mg) { "PARTIAL" }
    elseif ($a.MaxGen -ge 5) { "SHORT" }
    else { "ABORTED" }
  $rows += [pscustomobject]@{
    Tag=$a.Tag; Stage=$a.Stage; Seg=$a.Seg; MaxGen=$a.MaxGen; Need=$mg; Patience=$pt
    LastStag=$a.LastStag; Blocks=$a.Blocks; BadRows=$a.Bad
    FirstBICn=[math]::Round($a.FirstBICn,4); BestBICn=[math]::Round($a.BestBICn,4)
    EarlyStop=$a.EarlyStop; Verdict=$verdict; File=$a.LastFile
  }
}
$rows | Sort-Object @{e={[int]$_.Stage}}, Tag, Seg | Export-Csv "_gen_audit.csv" -NoTypeInformation
Write-Output ("parsed=" + $rows.Count)