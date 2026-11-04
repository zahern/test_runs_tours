raw = open(r'Z:\test_runs_tours\abm_safety\preprint\TAP_Poster_2026.tex', 'rb').read().decode('utf-8')
i = raw.find('\\block{Outcomes & findings}')
if i == -1:
    i = raw.find('\\block{Outcomes \\& findings}')
print(f'Found at {i}')
# Find the closing } of this block
j = i
brace = 0
for k in range(i, len(raw)):
    if raw[k] == '{':
        brace += 1
    elif raw[k] == '}':
        brace -= 1
        if brace == 0:
            j = k + 1
            break
print(f'Block end at {j}')
print(repr(raw[i:j][:500]))

new = r"""\block{Outcomes \& findings: what safety changes (\$\mathcal{M}_S-\mathcal{M}_0\$)}{
\centering
% Active travel +1.0 pp
\begin{tikzpicture}
  \node[draw=QUTGreen,line width=1.2pt,fill=QUTGreen!8,rounded corners=8pt,
        minimum width=11.6cm,minimum height=3.5cm,align=center] at (0,0)
        {{\color{QUTGreen}\bfseries\fontsize{34}{38}\selectfont $+1.0$\,pp}\\[3pt]
         {\small Active travel (walk $+$ bike)}};
  \node[draw=QUTNavy,line width=1.2pt,fill=QUTIce!45,rounded corners=8pt,
        minimum width=11.6cm,minimum height=3.5cm,align=center] at (13.2,0)
        {{\color{QUTNavy}\bfseries\fontsize{34}{38}\selectfont $-1.1$\,pp}\\[3pt]
         {\small Drive alone}};
\end{tikzpicture}

\vspace{0.15em}
{\footnotesize\color{QUTGrey}change in work-tour mode share,
$\mathcal{M}_S-\mathcal{M}_0$ on the full 3.82\,M synthetic population\par}

\vspace{0.3em}
% how-to-read box
\begin{tikzpicture}
  \node[draw=QUTAmber,line width=1pt,fill=QUTAmber!12,rounded corners=6pt,
        minimum width=25.2cm,minimum height=1.7cm,align=left,
        text width=24.4cm,font=\scriptsize] at (12.6,0)
        {{\color{QUTAmber}\bfseries How to read the pp numbers:}
         1\,pp (percentage point) = 1\% of all work tours.
         The two models run on \emph{identical machinery, identical population, identical seed} ---
         the only difference is that \$\mathcal{M}_S\$ adds crash-exposure terms
         to the mode and destination utilities, while \$\mathcal{M}_0\$ (ActivitySim-style baseline) does not.
         The \$\mathcal{M}_0\$ baseline shares are the reference; $\mathcal{M}_S$ shifts them.};
\end{tikzpicture}

\vspace{0.4em}
\raggedright
\begin{itemize}\itemsep0.14em
  \item Crash-exposure term \textbf{selected by BIC search} in Stage-5 mode choice
        for all three tour segments (work, education, non-mandatory).
  \item Destination crash coefficient (freely estimated): work $\mathbf{-1.00}$
        (SE $0.016$), education $\mathbf{-2.03}$ (SE $0.071$); work mode crash
        $\mathbf{-0.522}$ (SE $0.072$).
  \item Search-selected work mode crash coefficient $-0.127$ ($p=0.67$):
        BIC \emph{selection} vs.\ coefficient \emph{significance} reported separately.
  \item A $20\%$ crash-reduction counterfactual further raises active travel share.
\end{itemize}
}"""

if raw[i:j] in raw:
    raw2 = raw[:i] + new + raw[j:]
    with open(r'Z:\test_runs_tours\abm_safety\preprint\TAP_Poster_2026.tex', 'w', encoding='utf-8') as f:
        f.write(raw2)
    print('REPLACED OK')
else:
    print('OLD NOT FOUND')
    print('Search segment:', repr(raw[i:i+200]))