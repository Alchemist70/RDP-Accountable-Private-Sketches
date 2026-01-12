% Minimal stub of Springer svjour3 class for local compilation.
% This is not the official class; it simply wraps article to allow draft PDFs.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{svjour3}[2025/01/01 stub for local build]

% Forward any options to article
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass{article}

% Basic helpers used by manuscript
\newcommand{\journalname}[1]{\gdef\@journalname{#1}}
\providecommand{\qedsymbol}{$\square$}
\newcommand{\smartqed}{\renewcommand{\qedsymbol}{$\square$}}
\AtBeginDocument{\providecommand{\qed}{\qedsymbol}}

% Keep condensed font-ish defaults
\setlength{\textwidth}{6.0in}
\setlength{\textheight}{9.0in}
\setlength{\oddsidemargin}{0.25in}
\setlength{\evensidemargin}{0.25in}
\setlength{\topmargin}{-0.2in}

% Compatibility macros
\newcommand{\titlerunning}[1]{\gdef\@shorttitle{#1}}
\newcommand{\authorrunning}[1]{\gdef\@shortauthor{#1}}
\newcommand{\institute}[1]{\gdef\@institute{#1}}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

% Print minimal title block
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}\vskip1em
    {\large\@author\par}\vskip0.5em
    {\normalsize\@institute\par}
  \end{center}
  \vskip1em
  \ifx\@keywords\@empty\else{\small\textbf{Keywords:} \@keywords\par}\fi
}

\gdef\@institute{}
\gdef\@keywords{}

\endinput

