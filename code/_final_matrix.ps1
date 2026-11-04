$gen = Import-Csv "_gen_audit.csv"
$sh  = Import-Csv "_shard_audit.csv"
$exp = [ordered]@{
  "0"=@("all"); "1"=@("work","edu","nonmand"); "2"=@("start","end","budget");
  "3"=@("work","edu","nonmand"); "4"=@("work","edu","nonmand");
  "5"=@("work","edu","nonmand");
  "51"=@("work","edu","escort","shopping","othmaint","othdiscr","eat","social") }
$TAGS = @('qld_conv1','qld_fix1','qld_full40','qld_lccm3','lccm3_fixed')
$out=@()
foreach ($t in $TAGS) {
  foreach ($st in $exp.Keys) {
    foreach ($seg in $exp[$st]) {
      $g = $gen | Where-Object { $_.Tag -eq $t -and $_.Stage -eq $st -and $_.Seg -eq $seg } | Select-Object -First 1
      $s = $sh  | Where-Object { $_.Tag -eq $t -and $_.Stage -eq $st -and $_.Shard -eq $seg } | Select-Object -First 1
      $maxgen = if ($g) { $g.MaxGen } else { "-" }
      $need   = if ($g) { $g.Need }   else { "-" }
      $stag   = if ($g) { $g.LastStag } else { "-" }
      $bicn   = if ($g) { $g.BestBICn } else { "-" }
      $es     = if ($g) { $g.EarlyStop } else { "-" }
      $res    = if ($s) { $s.Results } else { "NO_DIR" }
      $fpkl   = if ($s) { $s.FailedPkl } else { "-" }
      $part   = if ($s) { $s.Partial } else { "-" }
      $model  = if (-not $s) { "NO" }
                elseif ($s.FailedPkl -eq "True" -or $s.Partial -eq "True") { "FAILED" }
                elseif ($s.Results -ne "True") { "NO_PKL" }
                elseif ($g -and $g.Verdict -like "CONVERGED*") { "VALID" }
                elseif (-not $g) { "NO_GA_EVID" }
                else { "UNCONVERGED" }
      $out += [pscustomobject]@{ Tag=$t; Stage=$st; Seg=$seg; MaxGen=$maxgen; Need=$need
        LastStag=$stag; BestBICn=$bicn; EarlyStop=$es; Model=$model; Res=$res; FailPkl=$fpkl }
    }
  }
}
$out | Export-Csv "_final_matrix.csv" -NoTypeInformation
$out | Where-Object {$_.Tag -eq 'qld_conv1'} | Format-Table Stage,Seg,MaxGen,Need,LastStag,BestBICn,EarlyStop,Model -AutoSize
Write-Output "`n=== MODEL STATUS TALLY per tag ==="
$out | Group-Object Tag,Model | Sort-Object Name | Format-Table Count,Name -AutoSize