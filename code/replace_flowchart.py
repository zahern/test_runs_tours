raw = open(r'Z:\test_runs_tours\abm_safety\preprint\TAP_Poster_2026.tex', 'rb').read().decode('utf-8')
i = raw.find('Full-width chain:')
j = raw.find('Footer band:')
block_end = raw.rfind('\n}\n\n', i, j)

old = raw[i:block_end+3]

new = r"""%==============================================================================
%  Full-width chain: two readable rows + "what the traveller weighs" icons
%==============================================================================
\block{From a crash skim to a daily plan: the chain, and where safety enters}{
\centering
\begin{tikzpicture}[font=\scriptsize,every node/.style={inner sep=1pt}]
  % geometry: full 78 cm block width, two rows, no overlaps
  \def\bw{14.35}   % stage-box width
  \def\bh{8.1}     % stage-box height
  \def\gap{0.85}   % horizontal gap between boxes
  \def\yrone{0}    % row 1 baseline
  \def\yrtwo{-11.2}

  % stage box: #1 x #2 y #3 stage #4 name #5 det #6 consider-icons
  \newcommand{\stageboxx}[6]{%
    \draw[fill=QUTIce!35,draw=QUTSky,line width=1.1pt,rounded corners=6pt]
      (#1,#2) rectangle (#1+\bw,#2+\bh);
    \node[anchor=north,font=\footnotesize\color{QUTGrey}]
      at ({#1+\bw/2},{#2+\bh-0.75}) {#3};
    \node[anchor=north,font=\small\bfseries\color{QUTNavy}]
      at ({#1+\bw/2},{#2+\bh-1.65}) {#4};
    \node[anchor=north,align=center,text width=12.8cm,font=\scriptsize\color{QUTGrey}]
      at ({#1+\bw/2},{#2+\bh-2.9}) {#5};
    % "what the traveller weighs" strip
    \draw[fill=white,draw=QUTSky!55,line width=0.7pt,rounded corners=10pt]
      (#1+0.8,#2+0.5) rectangle (#1+\bw-0.8,#2+2.0);
    \node[anchor=west,font=\tiny\itshape\color{QUTGrey}]
      at (#1+0.95,#2+1.28) {the traveller weighs};
    \pict{pperson}{(#1+3.4,#2+1.02)}{0.42}
    #6
  }
  \newcommand{\harrow}[2]{%
    \draw[-{Latex[length=3.4mm,width=2.8mm]},QUTGrey,line width=1.6pt]
      (#1,#2) -- (#1+\gap-0.16,#2);
  }

  % ROW 1: CDAP -> Frequency -> Budget -> Duration -> Placement
  \stageboxx{0}{\yrone}{Stage 0}{CDAP pattern}{free pattern: home / non-mand / mandatory}{
     \pict{phouse}{(1+\bw*0.30,\yrone+1.02)}{0.42}
     \pict{poffice}{(1+\bw*0.42,\yrone+1.02)}{0.42}}
  \harrow{\bw}{\yrone+\bh/2}
  \stageboxx{\bw+\gap}{\yrone}{Stage 1}{Frequency}{tours per purpose, incl.\ joint}{}
  \harrow{2*\bw+\gap}{\yrone+\bh/2}
  \stageboxx{2*\bw+2*\gap}{\yrone}{Stage 2}{Day budget}{hours available for activities}{}
  \harrow{3*\bw+3*\gap}{\yrone+\bh/2}
  \stageboxx{3*\bw+3*\gap}{\yrone}{Stage 3}{Duration}{minutes in each activity}{}
  \harrow{4*\bw+4*\gap}{\yrone+\bh/2}
  \stageboxx{4*\bw+4*\gap}{\yrone}{Stage 4}{Placement}{start times, 6 day periods}{}

  % wrap arrow row 1 -> row 2
  \draw[-{Latex[length=3.6mm,width=3.0mm]},QUTGrey,line width=1.6pt,rounded corners=9pt]
    (4*\bw+4*\gap+\bw/2,\yrone) --
    (4*\bw+4*\gap+\bw/2,\yrtwo+2.1) --
    (-1.05,\yrtwo+2.1) --
    (-1.05,\yrtwo+\bh+0.7) -- (\bw/2,\yrtwo+\bh+0.7) --
    (\bw/2,\yrtwo+\bh+0.15);

  % ROW 2: Mode -> Destination -> Stops -> Validation -> Outputs
  \stageboxx{0}{\yrtwo}{Stage 5a}{Tour mode}{nested logit over 8 modes}{
    \pict{pcar}{(0+4.6,\yrtwo+1.02)}{0.42}
    \pict{pbus}{(0+5.8,\yrtwo+1.02)}{0.42}
    \pict{pbike}{(0+7.0,\yrtwo+1.02)}{0.42}}
  \harrow{\bw}{\yrtwo+\bh/2}
  \stageboxx{\bw+\gap}{\yrtwo}{Stage 5b}{Tour destination}{371 SA2 zones $+$ mode logsum}{
    \pict{ppin}{(\bw+\gap+5.2,\yrtwo+1.02)}{0.42}}
  \harrow{2*\bw+\gap}{\yrtwo+\bh/2}
  \stageboxx{2*\bw+2*\gap}{\yrtwo}{Stage 5c}{Trip stops}{detour-feasible stops}{
    \pict{ppin}{(2*\bw+2*\gap+4.6,\yrtwo+1.02)}{0.42}
    \pict{phouse}{(2*\bw+2*\gap+5.8,\yrtwo+1.02)}{0.42}}
  \harrow{3*\bw+3*\gap}{\yrtwo+\bh/2}
  \stageboxx{3*\bw+3*\gap}{\yrtwo}{Stage 6}{Validation}{consistency checks $+$ repair}{}
  \harrow{4*\bw+4*\gap}{\yrtwo+\bh/2}
  \stageboxx{4*\bw+4*\gap}{\yrtwo}{Stage 7}{Outputs}{DynusT/MATSim plans $+$ OD}{
    \pict{pnet}{(4*\bw+4*\gap+5.2,\yrtwo+1.02)}{0.42}}

  % red safety hotspots
  \node[circle,fill=QUTRed,text=white,font=\small\bfseries,minimum size=0.85cm,
        inner sep=0pt] at (2*\bw+2*\gap+\bw-0.55,\yrone+\bh-0.55) {1};
  \node[circle,fill=QUTRed,text=white,font=\small\bfseries,minimum size=0.85cm,
        inner sep=0pt] at (\bw-0.55,\yrtwo+\bh-0.55) {2};
  \node[circle,fill=QUTRed,text=white,font=\small\bfseries,minimum size=0.85cm,
        inner sep=0pt] at (\bw+\gap+\bw-0.55,\yrtwo+\bh-0.55) {3};

  % what the traveller weighs: mode / destination notes
  \node[font=\tiny\color{QUTNavy},align=center] at (\bw+\gap+\bw/2,\yrtwo+0.08)
      {times, cost,\\ availability,\\ crash exposure};
  \node[font=\tiny\color{QUTNavy},align=center] at ({\bw+\gap+\bw/2+0.0},{\yrtwo+0.08})
      {size, distance, logsum,\\ crash risk};
\end{tikzpicture}

\vspace{0.35em}
\raggedright
{\small
\textcolor{QUTRed}{\textbf{1}}\; home-zone crash accessibility enters the
day-duration budget \quad$\cdot$\quad
\textcolor{QUTRed}{\textbf{2}}\; OD crash skim enters the mode utility
\quad$\cdot$\quad
\textcolor{QUTRed}{\textbf{3}}\; crash skim \emph{and} the mode logsum enter the
destination utility.\\[0.3em]
Every arrow is a separate model estimated on HHTS microdata, then applied to
all 3.82 million synthetic persons.\par}

\vspace{0.45em}
\centering
{\small
$V_{m} = \beta_t\,\mathrm{IVT}_m + \beta_c\,\mathrm{cost}_m
        + c_{\mathrm{crash,veh}}\,\widetilde{\mathrm{CRASH}}_{m} + \cdots$
\qquad\qquad
$V_{j} = \beta_s \ln S_j
        + c_{\mathrm{crash}}\,\widetilde{\ln(\mathrm{CRASH}_{jk}{+}1)}
        + \theta\,\mathrm{logsum}_m + \cdots$\par}
}"""

if old in raw:
    raw2 = raw[:i] + new + raw[block_end+3:]
    with open(r'Z:\test_runs_tours\abm_safety\preprint\TAP_Poster_2026.tex', 'w', encoding='utf-8') as f:
        f.write(raw2)
    print('REPLACED OK')
else:
    print('OLD NOT FOUND')
    # Debug: show first 200 chars of old and raw[i:j]
    print('old[:200]:', repr(old[:200]))
    print('raw[i:j][:200]:', repr(raw[i:j][:200]))