<#
Offline fallback renderer for the appendix convergence panels.

build_convergence_panels.py is the canonical builder and the one to use on the
HPC login node.  This script draws the same panels with GDI+ for the case where
the preprint tree is reachable (Z:) but no Python interpreter or cluster session
is.  It reads the audited histories listed in
abm_safety/preprint/convergence_audit.csv (written by the Python builder) and
writes abm_safety/preprint/figures/appendix/conv_s<stage>_<segment>.png.

Panel layout matches the Python builder:
  top     best-so-far BIC/n (bold), population mean (dashed, grey),
          generation best (thin lilac), population spread mean--worst (fill)
  bottom  population diversity
  header  stage/segment title and the verdict from convergence_audit.csv

Usage (from the repository root):
  powershell -ExecutionPolicy Bypass -File code/build_convergence_panels.ps1
#>
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

Add-Type -AssemblyName System.Drawing

$pre    = Join-Path $Root 'abm_safety\preprint'
$figdir = Join-Path $pre 'figures\appendix'
$audit  = Join-Path $pre 'convergence_audit.csv'
if (-not (Test-Path $audit)) { throw "missing audit: $audit" }
if (-not (Test-Path $figdir)) { New-Item -ItemType Directory -Path $figdir | Out-Null }

$STAGE_NAMES = @{ 0 = 'CDAP'; 1 = 'Activity frequency'; 2 = 'Day budget'; 3 = 'Activity duration'; 4 = 'Schedule placement' }
$SEG_LABEL = @{
    'all' = 'all persons'; 'nonmand' = 'non-mandatory (combined)';
    'othmaint' = 'other maintenance'; 'othdiscr' = 'other discretionary';
    'eat' = 'eat out'; 'start' = 'start of day'; 'end' = 'end of day'; 'budget' = 'day budget'
}

$COL_BEST = [System.Drawing.Color]::FromArgb(61, 58, 158)
$COL_GEN  = [System.Drawing.Color]::FromArgb(156, 153, 232)
$COL_MEAN = [System.Drawing.Color]::FromArgb(102, 102, 102)
$COL_FILL = [System.Drawing.Color]::FromArgb(26, 201, 138, 58)
$COL_GRID = [System.Drawing.Color]::FromArgb(60, 0, 0, 0)
$COL_AXIS = [System.Drawing.Color]::FromArgb(90, 0, 0, 0)

$F_TITLE  = New-Object System.Drawing.Font('Segoe UI', 23, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
$F_LABEL  = New-Object System.Drawing.Font('Segoe UI', 19, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$F_TICK   = New-Object System.Drawing.Font('Segoe UI', 16, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
$F_SMALL  = New-Object System.Drawing.Font('Segoe UI', 15, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)

function New-Pen([System.Drawing.Color]$c, [float]$w, [string]$dash = 'Solid') {
    $p = New-Object System.Drawing.Pen($c, $w)
    $p.DashStyle = $dash
    $p.StartCap = 'Round'; $p.EndCap = 'Round'
    return $p
}

function Get-Num($v) {
    $d = 0.0
    if ([double]::TryParse([string]$v, [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture, [ref]$d)) { return $d }
    return [double]::NaN
}

$rows = Import-Csv -Path $audit
$made = 0
foreach ($row in $rows) {
    $sid = [int]$row.stage
    $seg = $row.segment
    $histPath = Join-Path (Join-Path $Root 'code') $row.history
    if (-not (Test-Path $histPath)) { Write-Warning "missing history: $histPath"; continue }
    $hist = @(Import-Csv -Path $histPath)
    # A pooled history file carries several segments; keep this one's rows.
    if ($hist.Count -gt 0 -and $hist[0].PSObject.Properties.Name -contains 'segment') {
        $names = @($hist | Select-Object -ExpandProperty segment -Unique)
        if ($names.Count -gt 1) {
            $want = $seg.ToLower()
            $filtered = @($hist | Where-Object { $_.segment.ToLower().Replace('-', '').Replace('_', '').Contains($want) })
            if ($filtered.Count -ge 2) { $hist = $filtered }
        }
    }
    if ($hist.Count -lt 1) { Write-Warning "empty history: $histPath"; continue }

    $gen  = @(); $best = @(); $bsf = @(); $mean = @(); $worst = @(); $div = @()
    for ($i = 0; $i -lt $hist.Count; $i++) {
        $r = $hist[$i]
        $gen  += [double](Get-Num $r.generation)
        $best += -(Get-Num $r.best_fitness)
        $bsf  += -(Get-Num $r.best_so_far)
        $mean += -(Get-Num $r.mean_fitness)
        $worst += -(Get-Num $r.worst_fitness)
        $div  += (Get-Num $r.diversity)
    }
    $n = $hist.Count

    $core = @($bsf + $mean | Where-Object { -not [double]::IsNaN($_) })
    if ($core.Count -lt 1) { Write-Warning "no finite BIC values: $histPath"; continue }
    $lo = ($core | Measure-Object -Minimum).Minimum
    $hi = ($core | Measure-Object -Maximum).Maximum
    $pad = [math]::Max(0.08 * ($hi - $lo), 0.01)
    $y0 = $lo - $pad; $y1 = $hi + $pad
    if ($y1 -le $y0) { $y1 = $y0 + 0.01 }

    $W = 1140; $H = 660
    $bmp = New-Object System.Drawing.Bitmap($W, $H)
    $bmp.SetResolution(150, 150)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)

    $L = 108; $R = 1092
    $T0 = 96; $B0 = 470          # top panel plot box
    $T1 = 512; $B1 = 612         # bottom panel plot box

    function XPos($i) { if ($n -le 1) { return ($L + $R) / 2 } return $L + ($R - $L) * ($i / ($n - 1)) }
    function YTopPos($v) { return $B0 - ($B0 - $T0) * (($v - $y0) / ($y1 - $y0)) }

    # title + verdict
    $title = 'Stage ' + $sid + '  ' + [char]0x00B7 + '  ' + $STAGE_NAMES[$sid] + '  ' + [char]0x00B7 + '  ' + ($(if ($SEG_LABEL.ContainsKey($seg)) { $SEG_LABEL[$seg] } else { $seg }))
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = 'Center'
    $white = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Black)
    $g.DrawString($title, $F_TITLE, $white, ($W / 2), 22, $fmt)
    $conv = ([string]$row.converged).ToLower() -eq 'true'
    $vc = if ($conv) { [System.Drawing.Color]::FromArgb(27, 127, 58) } else { [System.Drawing.Color]::FromArgb(179, 38, 30) }
    $vbrush = New-Object System.Drawing.SolidBrush($vc)
    $g.DrawString([string]$row.verdict, $F_SMALL, $vbrush, ($W / 2), 60, $fmt)

    # y grid + ticks (top)
    $gridPen = New-Pen $COL_GRID 1
    for ($k = 0; $k -le 4; $k++) {
        $v = $y0 + ($y1 - $y0) * $k / 4
        $yy = YTopPos $v
        $g.DrawLine($gridPen, $L, $yy, $R, $yy)
        $lab = $v.ToString('0.000', [System.Globalization.CultureInfo]::InvariantCulture)
        $sz = $g.MeasureString($lab, $F_TICK)
        $g.DrawString($lab, $F_TICK, $white, ($L - 10 - $sz.Width), ($yy - $sz.Height / 2))
    }

    # Clip every data mark to the axes: the spread polygon and the mean can
    # legitimately leave the chosen y-window, and unclipped marks would paint
    # over the title and the verdict line.
    $g.SetClip((New-Object System.Drawing.Rectangle($L, $T0, ($R - $L), ($B0 - $T0))))

    # population spread.  Values outside the y-window are clamped to its edge
    # before the polygon is built: a self-intersecting polygon would otherwise
    # leave holes in the fill wherever the worst traveller overshoots the top.
    $pts = New-Object System.Collections.Generic.List[System.Drawing.PointF]
    for ($i = 0; $i -lt $n; $i++) {
        if ([double]::IsNaN($mean[$i]) -or [double]::IsNaN($worst[$i])) { continue }
        $pts.Add((New-Object System.Drawing.PointF((XPos $i), (YTopPos ([math]::Min($y1, [math]::Max($y0, $mean[$i])))))))
    }
    $tmp = @()
    for ($i = $n - 1; $i -ge 0; $i--) {
        if ([double]::IsNaN($mean[$i]) -or [double]::IsNaN($worst[$i])) { continue }
        $tmp += (New-Object System.Drawing.PointF((XPos $i), (YTopPos ([math]::Min($y1, [math]::Max($y0, $worst[$i]))))))
    }
    foreach ($p in $tmp) { $pts.Add($p) }
    if ($pts.Count -ge 3) {
        $fill = New-Object System.Drawing.SolidBrush($COL_FILL)
        $g.FillPolygon($fill, $pts.ToArray())
        $fill.Dispose()
    }

    # The logs record generation_best == best_so_far on every row (the winner
    # is written after the incumbent update), so a separate generation-best
    # series would sit exactly under the bold line; it is not drawn.
    $pMean = New-Pen $COL_MEAN 2.4 'Dash'
    $pBest = New-Pen $COL_BEST 5.0
    for ($i = 1; $i -lt $n; $i++) {
        $g.DrawLine($pMean, (XPos ($i - 1)), (YTopPos $mean[$i - 1]), (XPos $i), (YTopPos $mean[$i]))
        $g.DrawLine($pBest, (XPos ($i - 1)), (YTopPos $bsf[$i - 1]),  (XPos $i), (YTopPos $bsf[$i]))
    }
    if ($n -le 5) {
        $bBest = New-Object System.Drawing.SolidBrush($COL_BEST)
        $bMean = New-Object System.Drawing.SolidBrush($COL_MEAN)
        for ($i = 0; $i -lt $n; $i++) {
            $g.FillEllipse($bBest, ((XPos $i) - 4), ((YTopPos $bsf[$i]) - 4), 8, 8)
            $g.FillEllipse($bMean, ((XPos $i) - 3), ((YTopPos $mean[$i]) - 3), 6, 6)
        }
        $bBest.Dispose(); $bMean.Dispose()
    }
    $pMean.Dispose(); $pBest.Dispose()
    $g.ResetClip()

    # axes box (top)
    $axisPen = New-Pen $COL_AXIS 1.4
    $g.DrawRectangle($axisPen, $L, $T0, ($R - $L), ($B0 - $T0))

    # y-axis label (rotated)
    $state = $g.Save()
    $g.TranslateTransform(26, ($T0 + $B0) / 2)
    $g.RotateTransform(-90)
    $g.DrawString('BIC / n   (lower is better)', $F_LABEL, $white, 0, 0, $fmt)
    $g.Restore($state)

    # legend (top right, two rows inside the axes)
    $items = @(
        @{ c = $COL_BEST; t = 'best-so-far'; w = 5.0; dash = 'Solid' },
        @{ c = $COL_MEAN; t = 'population mean'; w = 2.4; dash = 'Dash' },
        @{ c = $COL_FILL; t = 'population spread (mean--worst)'; w = 9.0; dash = 'Solid' }
    )
    $lx = $R - 300; $ly = $T0 + 12
    $legendBg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(232, 255, 255, 255))
    $g.FillRectangle($legendBg, ($lx - 10), ($ly - 6), 292, 80)
    $legendBg.Dispose()
    foreach ($it in $items) {
        if ($it.t -like 'population spread*') {
            $sw = New-Object System.Drawing.SolidBrush($COL_FILL)
            $g.FillRectangle($sw, $lx, ($ly + 2), 34, 14)
            $sw.Dispose()
            $edge = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(90, 201, 138, 58), 1)
            $g.DrawRectangle($edge, $lx, ($ly + 2), 34, 14)
            $edge.Dispose()
        } else {
            $pen = New-Pen $it.c $it.w $it.dash
            $g.DrawLine($pen, $lx, ($ly + 9), ($lx + 34), ($ly + 9))
            $pen.Dispose()
        }
        $g.DrawString($it.t, $F_SMALL, $white, ($lx + 42), ($ly - 2))
        $ly += 22
    }

    # x ticks: the two panels share the x scale, so label them only under the
    # diversity panel (as the Python builder does with sharex).
    $ticks = 5

    # bottom panel: diversity
    $dmax = 0.0
    foreach ($d in $div) { if (-not [double]::IsNaN($d) -and $d -gt $dmax) { $dmax = $d } }
    if ($dmax -le 0) { $dmax = 1.0 }
    $d1 = $dmax * 1.08
    function YDivPos($v) { return $B1 - ($B1 - $T1) * ($v / $d1) }
    for ($k = 0; $k -le 2; $k++) {
        $v = $d1 * $k / 2
        $yy = YDivPos $v
        $g.DrawLine($gridPen, $L, $yy, $R, $yy)
        $lab = $v.ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture)
        $sz = $g.MeasureString($lab, $F_TICK)
        $g.DrawString($lab, $F_TICK, $white, ($L - 10 - $sz.Width), ($yy - $sz.Height / 2))
    }
    $pDiv = New-Pen ([System.Drawing.Color]::FromArgb(79, 109, 122)) 1.8
    $g.SetClip((New-Object System.Drawing.Rectangle($L, $T1, ($R - $L), ($B1 - $T1))))
    for ($i = 1; $i -lt $n; $i++) {
        $g.DrawLine($pDiv, (XPos ($i - 1)), (YDivPos $div[$i - 1]), (XPos $i), (YDivPos $div[$i]))
    }
    $pDiv.Dispose()
    $g.ResetClip()
    $g.DrawRectangle($axisPen, $L, $T1, ($R - $L), ($B1 - $T1))
    $state = $g.Save()
    $g.TranslateTransform(26, ($T1 + $B1) / 2)
    $g.RotateTransform(-90)
    $g.DrawString('diversity', $F_LABEL, $white, 0, 0, $fmt)
    $g.Restore($state)
    $g.DrawString('Generation', $F_LABEL, $white, ($W / 2), ($B1 + 18), $fmt)
    for ($k = 0; $k -le $ticks; $k++) {
        $idx = [math]::Round(($n - 1) * $k / $ticks)
        $xx = XPos $idx
        $g.DrawLine($axisPen, $xx, $B1, $xx, ($B1 + 5))
        $lab = [string][int]$gen[$idx]
        $sz = $g.MeasureString($lab, $F_TICK)
        $g.DrawString($lab, $F_TICK, $white, ($xx - $sz.Width / 2), ($B1 + 6))
    }

    $out = Join-Path $figdir ('conv_s' + $sid + '_' + $seg + '.png')
    $g.Dispose()
    $bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    $made++
    Write-Host ('[ok] s{0}/{1,-9} gens={2,4} {3}' -f $sid, $seg, $n, $(if ($conv) { 'CONVERGED' } else { 'NOT CONVERGED' }))
}
Write-Host "wrote $made panels to $figdir"
