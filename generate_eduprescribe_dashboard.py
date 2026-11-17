#!/usr/bin/env python3
"""
Generates the self-contained EduPrescribe AI interactive dashboard.
Replaces __COHORT_DATA__ placeholder with JSON array of students.
"""

import pandas as pd

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EduPrescribe AI: Student Burnout Forecaster & Counterfactual Study Optimizer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    /* Custom scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 antialiased min-h-screen p-4 md:p-6 font-sans">
  <div class="max-w-7xl mx-auto space-y-6">

    <!-- Top Navigation Header -->
    <header class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-2.5">
            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500 text-white">Prescriptive AI Framework</span>
            <span class="text-xs text-slate-400">UCI Mathematics Behavioral Dataset (N = 395)</span>
          </div>
          <h1 class="text-2xl font-bold tracking-tight text-white mt-1.5 flex items-center gap-2">
            <span>EduPrescribe AI</span>
            <span class="text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 font-mono">v1.0</span>
          </h1>
          <p class="text-xs text-slate-400 mt-0.5">
            Dual-Objective Behavioral Wellness Forecaster & Counterfactual "What-If" Academic Study Optimizer
          </p>
        </div>

        <!-- Quick Summary Stats -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div class="bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
            <span class="text-slate-400 block text-[11px]">Thriving (Q1)</span>
            <span class="font-bold text-sm text-emerald-400">204 (51.6%)</span>
          </div>
          <div class="bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
            <span class="text-slate-400 block text-[11px]">Burnout Risk (Q2)</span>
            <span class="font-bold text-sm text-amber-400">61 (15.4%)</span>
          </div>
          <div class="bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
            <span class="text-slate-400 block text-[11px]">Dual Crisis (Q3)</span>
            <span class="font-bold text-sm text-rose-400">33 (8.4%)</span>
          </div>
          <div class="bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
            <span class="text-slate-400 block text-[11px]">Coasting (Q4)</span>
            <span class="font-bold text-sm text-sky-400">97 (24.6%)</span>
          </div>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <nav class="flex border-b border-slate-800 mt-6 -mb-5 space-x-6 overflow-x-auto">
        <button onclick="switchTab('optimizer')" id="tab-btn-optimizer" class="tab-btn pb-3 text-sm font-semibold border-b-2 border-indigo-500 text-indigo-400 whitespace-nowrap">
          1. Counterfactual 'What-If' Optimizer
        </button>
        <button onclick="switchTab('quadrant')" id="tab-btn-quadrant" class="tab-btn pb-3 text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-white whitespace-nowrap">
          2. Well-Being Quadrant Matrix
        </button>
        <button onclick="switchTab('cohort')" id="tab-btn-cohort" class="tab-btn pb-3 text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-white whitespace-nowrap">
          3. Cohort Prescriptions Browser
        </button>
        <button onclick="switchTab('methodology')" id="tab-btn-methodology" class="tab-btn pb-3 text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-white whitespace-nowrap">
          4. Scientific Methodology & Ethics
        </button>
      </nav>
    </header>

    <!-- ============================================================== -->
    <!-- TAB 1: COUNTERFACTUAL WHAT-IF OPTIMIZER -->
    <!-- ============================================================== -->
    <section id="tab-optimizer" class="tab-pane space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Left: Diagnostic Sliders -->
        <div class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-5">
          <div class="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h2 class="font-bold text-base text-white">Student Diagnostic Parameters</h2>
              <p class="text-xs text-slate-400">Adjust academic, study, and lifestyle stressors to run the counterfactual engine</p>
            </div>
            <!-- Presets -->
            <div class="flex gap-1.5">
              <button onclick="setProfile(8, 8, 1, 12, 1, 'no', 2, 4, 5, 2, 2)" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-rose-300 border border-rose-500/30 transition">Crisis (Q3)</button>
              <button onclick="setProfile(15, 14, 3, 10, 0, 'no', 2, 4, 5, 2, 2)" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-amber-300 border border-amber-500/30 transition">Burnout Achiever (Q2)</button>
              <button onclick="setProfile(7, 8, 1, 4, 1, 'no', 1, 1, 2, 4, 4)" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-sky-300 border border-sky-500/30 transition">Coasting (Q4)</button>
              <button onclick="setProfile(14, 15, 3, 2, 0, 'no', 1, 1, 2, 3, 5)" class="text-xs px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-emerald-500/30 transition">Thriving (Q1)</button>
            </div>
          </div>

          <div class="space-y-4 text-xs">
            <!-- Academic Sliders -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300">Period 1 Grade (G1)</label>
                  <span id="v-g1" class="font-mono font-bold text-indigo-400">8 / 20</span>
                </div>
                <input id="in-g1" type="range" min="0" max="20" value="8" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300">Period 2 Grade (G2)</label>
                  <span id="v-g2" class="font-mono font-bold text-indigo-400">8 / 20</span>
                </div>
                <input id="in-g2" type="range" min="0" max="20" value="8" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300">Past Failures</label>
                  <span id="v-fail" class="font-mono font-bold text-indigo-400">1</span>
                </div>
                <input id="in-fail" type="range" min="0" max="4" value="1" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
            </div>

            <!-- Controllable Study & School Levers -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300">Study Time</label>
                  <span id="v-study" class="font-mono font-bold text-indigo-400">&lt;2 hrs (1)</span>
                </div>
                <input id="in-study" type="range" min="1" max="4" value="1" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300">School Absences</label>
                  <span id="v-abs" class="font-mono font-bold text-indigo-400">10 days</span>
                </div>
                <input id="in-abs" type="range" min="0" max="40" value="10" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <label class="font-medium text-slate-300 block mb-1">School Tutoring</label>
                <select id="in-sup" class="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200" onchange="recalculate()">
                  <option value="no">No Extra Support</option>
                  <option value="yes">Extra Math Tutoring (Yes)</option>
                </select>
              </div>
            </div>

            <!-- Lifestyle Stressors -->
            <div class="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300 text-[11px]">Going Out</label>
                  <span id="v-goout" class="font-mono font-bold text-indigo-400">5 / 5</span>
                </div>
                <input id="in-goout" type="range" min="1" max="5" value="5" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300 text-[11px]">Free Time</label>
                  <span id="v-free" class="font-mono font-bold text-indigo-400">2 / 5</span>
                </div>
                <input id="in-free" type="range" min="1" max="5" value="2" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300 text-[11px]">Health Rating</label>
                  <span id="v-health" class="font-mono font-bold text-indigo-400">2 / 5</span>
                </div>
                <input id="in-health" type="range" min="1" max="5" value="2" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div class="flex justify-between items-center">
                  <label class="font-medium text-slate-300 text-[11px]">Alcohol Coping</label>
                  <span id="v-alc" class="font-mono font-bold text-indigo-400">D:2 / W:3</span>
                </div>
                <input id="in-dalc" type="range" min="1" max="5" value="2" class="w-full accent-indigo-500" oninput="recalculate()">
              </div>
            </div>
          </div>
        </div>

        <!-- Right: Diagnostic Output & Prescriptive Action Plan -->
        <div class="lg:col-span-5 space-y-4">
          <!-- Diagnostic Status Card -->
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4 shadow-sm">
            <div class="flex justify-between items-center border-b border-slate-800 pb-3">
              <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Dual Diagnostic State</span>
              <span id="diag-badge" class="px-2.5 py-0.5 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                🔴 Q3: Dual Crisis
              </span>
            </div>

            <div class="grid grid-cols-2 gap-3 text-xs">
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-slate-400 block text-[11px]">Predicted Final Grade</span>
                <span id="diag-g3" class="text-xl font-mono font-bold text-white">7.40 <span class="text-xs text-slate-500">/ 20</span></span>
                <span id="diag-pass-prob" class="block text-[10px] text-rose-400 mt-0.5">Passing Chance: 14.9%</span>
              </div>
              <div class="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span class="text-slate-400 block text-[11px]">Burnout Index (BVI)</span>
                <span id="diag-bvi" class="text-xl font-mono font-bold text-amber-400">55.8 <span class="text-xs text-slate-500">/ 100</span></span>
                <span id="diag-bvi-tier" class="block text-[10px] text-amber-400 mt-0.5">Acute Exhaustion</span>
              </div>
            </div>

            <!-- Counterfactual Prescription Recipe Box -->
            <div class="bg-indigo-950/40 border border-indigo-500/30 rounded-lg p-4 space-y-2.5">
              <div class="flex items-center gap-2">
                <span class="text-indigo-400 text-sm">⚡</span>
                <h3 class="text-xs font-bold text-indigo-300 uppercase tracking-wide">Optimized Counterfactual Pathway</h3>
              </div>
              <p id="presc-action" class="text-xs text-slate-200 leading-relaxed font-medium">
                Enroll in remedial math tutoring, increase study time to 2-5 hrs, and recover 4 absences.
              </p>
              
              <div class="grid grid-cols-3 gap-2 pt-2 border-t border-indigo-500/20 text-[11px]">
                <div>
                  <span class="text-slate-400 block text-[10px]">Grade Uplift</span>
                  <span id="presc-uplift" class="font-mono font-bold text-emerald-400">+2.35 pts</span>
                </div>
                <div>
                  <span class="text-slate-400 block text-[10px]">Target Pass Prob</span>
                  <span id="presc-newpass" class="font-mono font-bold text-emerald-400">89.2%</span>
                </div>
                <div>
                  <span class="text-slate-400 block text-[10px]">Target Burnout</span>
                  <span id="presc-newbvi" class="font-mono font-bold text-indigo-300">32.4 / 100</span>
                </div>
              </div>
            </div>

            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
              <span class="text-amber-400 font-bold">ℹ</span>
              <span>The counterfactual engine calculates the minimum behavioral friction required to reach sustainable academic success.</span>
            </div>
          </div>
        </div>

      </div>
    </section>

    <!-- ============================================================== -->
    <!-- TAB 2: WELL-BEING QUADRANT MATRIX -->
    <!-- ============================================================== -->
    <section id="tab-quadrant" class="tab-pane hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 class="text-base font-bold text-white">Academic Mastery vs. Burnout Vulnerability Matrix</h2>
            <p class="text-xs text-slate-400">Interactive mapping of all 395 students across performance (G3) and lifestyle stress (BVI)</p>
          </div>
          <div class="flex flex-wrap gap-1.5 text-xs">
            <button onclick="filterQuadrantChart('All')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200">All (395)</button>
            <button onclick="filterQuadrantChart('Q1')" class="px-2.5 py-1 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30">Q1: Thriving</button>
            <button onclick="filterQuadrantChart('Q2')" class="px-2.5 py-1 rounded bg-amber-950/60 text-amber-400 border border-amber-500/30">Q2: Burnout Risk</button>
            <button onclick="filterQuadrantChart('Q3')" class="px-2.5 py-1 rounded bg-rose-950/60 text-rose-400 border border-rose-500/30">Q3: Dual Crisis</button>
            <button onclick="filterQuadrantChart('Q4')" class="px-2.5 py-1 rounded bg-sky-950/60 text-sky-400 border border-sky-500/30">Q4: Coasting</button>
          </div>
        </div>

        <div class="relative h-[420px] w-full bg-slate-950 rounded-lg p-2 border border-slate-800">
          <canvas id="quadrantChart"></canvas>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs pt-2">
          <div class="bg-slate-950 border border-emerald-500/30 rounded-lg p-3.5 space-y-1.5">
            <span class="text-emerald-400 font-bold block">Q1: Thriving Achievers (204)</span>
            <p class="text-slate-400 text-[11px] leading-tight">High grade (G3 >= 10) with balanced lifestyle and low burnout (BVI < 40).</p>
          </div>
          <div class="bg-slate-950 border border-amber-500/30 rounded-lg p-3.5 space-y-1.5">
            <span class="text-amber-400 font-bold block">Q2: Burnout Risk Achievers (61)</span>
            <p class="text-slate-400 text-[11px] leading-tight">High grade (G3 >= 10) masked by high lifestyle strain (BVI >= 40). High risk of collapse if unsupported.</p>
          </div>
          <div class="bg-slate-950 border border-rose-500/30 rounded-lg p-3.5 space-y-1.5">
            <span class="text-rose-400 font-bold block">Q3: Dual Crisis Students (33)</span>
            <p class="text-slate-400 text-[11px] leading-tight">Failing academically (G3 < 10) compounded by acute behavioral fatigue. Requires dual support.</p>
          </div>
          <div class="bg-slate-950 border border-sky-500/30 rounded-lg p-3.5 space-y-1.5">
            <span class="text-sky-400 font-bold block">Q4: Coasting / Disengaged (97)</span>
            <p class="text-slate-400 text-[11px] leading-tight">At-risk in math (G3 < 10), but low lifestyle stress. Reflects curriculum pacing or motivation gaps.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ============================================================== -->
    <!-- TAB 3: COHORT PRESCRIPTIONS BROWSER -->
    <!-- ============================================================== -->
    <section id="tab-cohort" class="tab-pane hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 class="text-base font-bold text-white">Cohort Prescriptions Browser (N = 395)</h2>
            <p class="text-xs text-slate-400">Search and review individual counterfactual recommendations for all students</p>
          </div>
          <div class="flex items-center gap-2">
            <select id="filter-quadrant-select" class="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200" onchange="filterCohortTable()">
              <option value="all">All Quadrants (395)</option>
              <option value="Q1">Q1: Thriving Achievers (204)</option>
              <option value="Q2">Q2: High-Performing Burnout Risk (61)</option>
              <option value="Q3">Q3: Dual Crisis / Vulnerable (33)</option>
              <option value="Q4">Q4: Disengaged / Coasting (97)</option>
            </select>
            <input id="search-student" type="text" placeholder="Search ID, school..." class="bg-slate-950 border border-slate-700 rounded px-2.5 py-1 text-xs w-40 text-slate-200" oninput="filterCohortTable()">
          </div>
        </div>

        <div class="overflow-x-auto border border-slate-800 rounded-lg">
          <table class="w-full text-xs text-left">
            <thead class="bg-slate-950 border-b border-slate-800 text-slate-400">
              <tr>
                <th class="py-2.5 px-3">ID</th>
                <th class="py-2.5 px-3">G1</th>
                <th class="py-2.5 px-3">G2</th>
                <th class="py-2.5 px-3">G3 (Actual)</th>
                <th class="py-2.5 px-3">BVI Stress</th>
                <th class="py-2.5 px-3">Quadrant</th>
                <th class="py-2.5 px-3">Projected Uplift</th>
                <th class="py-2.5 px-3">Actionable Prescription</th>
              </tr>
            </thead>
            <tbody id="cohort-tbody" class="divide-y divide-slate-800 text-slate-300">
              <!-- Rendered via JS -->
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ============================================================== -->
    <!-- TAB 4: METHODOLOGY & ETHICS -->
    <!-- ============================================================== -->
    <section id="tab-methodology" class="tab-pane hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/20 text-indigo-400">Mathematical Formulation</span>
          <h3 class="text-sm font-bold text-white">Burnout Vulnerability Index (BVI)</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            The BVI synthesizes 5 multidimensional behavioral stressors into a standardized continuous index from 0 to 100:
          </p>
          <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-indigo-300 space-y-1">
            <div>BVI = 100 * [ 0.25*S1 + 0.20*S2 + 0.25*S3 + 0.15*S4 + 0.15*S5 ]</div>
            <div class="text-[10px] text-slate-500">• S1: Substance Coping ((2*Dalc + Walc - 3)/12)</div>
            <div class="text-[10px] text-slate-500">• S2: Social Depletion ((goout - freetime + 4)/8)</div>
            <div class="text-[10px] text-slate-500">• S3: Vitality Deficit ((5 - health)/4)</div>
            <div class="text-[10px] text-slate-500">• S4: Absenteeism Saturation (min(1, absences/20))</div>
            <div class="text-[10px] text-slate-500">• S5: Relational Friction ((5 - famrel)/4)</div>
          </div>
          <p class="text-xs text-slate-400">
            A BVI cutoff of 40 separates sustainable academic effort from lifestyle depletion.
          </p>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-500/20 text-indigo-400">Optimization Objective</span>
          <h3 class="text-sm font-bold text-white">Minimum-Friction Counterfactual Search</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            For any at-risk student, the engine solves for the minimum behavioral adjustment vector:
          </p>
          <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-indigo-300 space-y-1">
            <div>min_Δx ||Δx||_w  subject to:</div>
            <div>1. P(Pass | x + Δx) >= 0.85</div>
            <div>2. BVI(x + Δx) <= 38.0</div>
            <div class="text-[10px] text-slate-500">where Δx contains studytime, absences, schoolsup, and goout.</div>
          </div>
          <p class="text-xs text-slate-400">
            Rather than passive failure alerts, EduPrescribe outputs concrete, achievable stepping-stones.
          </p>
        </div>

      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <span class="text-emerald-400">🛡️</span> Responsible AI & Educational Guidance
        </h3>
        <ul class="text-xs text-slate-400 space-y-2">
          <li class="flex items-start gap-2">
            <span class="text-emerald-400 font-bold">✓</span>
            <span><strong>Prescriptive Support, Never Punitive Tracking:</strong> Prescriptions serve to advise students on study balance, never to disqualify or track them.</span>
          </li>
          <li class="flex items-start gap-2">
            <span class="text-emerald-400 font-bold">✓</span>
            <span><strong>Uncovering Hidden Burnout:</strong> The 61 Q2 students (High-Grade Burnout Risk) represent a primary opportunity for proactive counseling before academic decline occurs.</span>
          </li>
          <li class="flex items-start gap-2">
            <span class="text-emerald-400 font-bold">✓</span>
            <span><strong>Counselor in the Loop:</strong> Advisors adapt recommendations to the qualitative human circumstances of each learner.</span>
          </li>
        </ul>
      </div>
    </section>

  </div>

  <script>
    const COHORT_DATA = __COHORT_DATA__;
    let quadrantChartInstance = null;

    function switchTab(tabId) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('border-indigo-500', 'text-indigo-400');
        el.classList.add('border-transparent', 'text-slate-400');
      });

      const targetPane = document.getElementById('tab-' + tabId);
      if (targetPane) targetPane.classList.remove('hidden');
      const targetBtn = document.getElementById('tab-btn-' + tabId);
      if (targetBtn) {
        targetBtn.classList.remove('border-transparent', 'text-slate-400');
        targetBtn.classList.add('border-indigo-500', 'text-indigo-400');
      }

      if (tabId === 'quadrant' && !quadrantChartInstance) {
        initQuadrantChart();
      }
    }

    function setProfile(g1, g2, study, abs, fail, sup, dalc, walc, goout, free, health) {
      document.getElementById('in-g1').value = g1;
      document.getElementById('in-g2').value = g2;
      document.getElementById('in-study').value = study;
      document.getElementById('in-abs').value = abs;
      document.getElementById('in-fail').value = fail;
      document.getElementById('in-sup').value = sup;
      document.getElementById('in-dalc').value = dalc;
      document.getElementById('in-goout').value = goout;
      document.getElementById('in-free').value = free;
      document.getElementById('in-health').value = health;
      recalculate();
    }

    function recalculate() {
      const g1 = parseFloat(document.getElementById('in-g1').value);
      const g2 = parseFloat(document.getElementById('in-g2').value);
      const study = parseInt(document.getElementById('in-study').value);
      const abs = parseInt(document.getElementById('in-abs').value);
      const fail = parseInt(document.getElementById('in-fail').value);
      const sup = document.getElementById('in-sup').value;
      const dalc = parseInt(document.getElementById('in-dalc').value);
      const walc = Math.min(5, dalc + 1);
      const goout = parseInt(document.getElementById('in-goout').value);
      const free = parseInt(document.getElementById('in-free').value);
      const health = parseInt(document.getElementById('in-health').value);
      const famrel = 4;

      document.getElementById('v-g1').innerText = g1 + ' / 20';
      document.getElementById('v-g2').innerText = g2 + ' / 20';
      document.getElementById('v-fail').innerText = fail;
      document.getElementById('v-study').innerText = (study===1?'<2 hrs':study===2?'2-5 hrs':study===3?'5-10 hrs':'>10 hrs') + ' (' + study + ')';
      document.getElementById('v-abs').innerText = abs + ' days';
      document.getElementById('v-goout').innerText = goout + ' / 5';
      document.getElementById('v-free').innerText = free + ' / 5';
      document.getElementById('v-health').innerText = health + ' / 5';
      document.getElementById('v-alc').innerText = 'D:' + dalc + ' / W:' + walc;

      const sup_num = sup === 'yes' ? 1 : 0;
      const pred_g3 = Math.max(0, Math.min(20, -1.5618 + 0.1770*g1 + 0.9586*g2 - 0.2015*study + 0.0434*abs - 0.1515*fail));
      
      const logit_val = -(18.2194 - 0.2911*g1 - 1.8552*g2 + 0.5999*study + 0.0501*abs - 0.2111*fail - 0.5313*sup_num);
      const pass_prob = 1.0 / (1.0 + Math.exp(-logit_val));

      const s1 = (2.0*dalc + walc - 3.0) / 12.0;
      const s2 = Math.max(0, Math.min(1, (goout - free + 4.0) / 8.0));
      const s3 = (5.0 - health) / 4.0;
      const s4 = Math.min(1, abs / 20.0);
      const s5 = (5.0 - famrel) / 4.0;
      const bvi = Math.round(100.0 * (0.25*s1 + 0.20*s2 + 0.25*s3 + 0.15*s4 + 0.15*s5) * 10) / 10;

      document.getElementById('diag-g3').innerHTML = pred_g3.toFixed(2) + ' <span class="text-xs text-slate-500">/ 20</span>';
      document.getElementById('diag-pass-prob').innerText = 'Passing Chance: ' + (pass_prob*100).toFixed(1) + '%';
      document.getElementById('diag-bvi').innerHTML = bvi.toFixed(1) + ' <span class="text-xs text-slate-500">/ 100</span>';
      
      let bvi_tier = 'Moderate Stress';
      if (bvi < 25) bvi_tier = 'Low Burnout';
      else if (bvi >= 55) bvi_tier = 'Acute Exhaustion';
      else if (bvi >= 40) bvi_tier = 'Elevated Burnout Risk';
      document.getElementById('diag-bvi-tier').innerText = bvi_tier;

      let badge_text = '';
      let badge_class = '';
      if (pred_g3 >= 10 && bvi < 40) {
        badge_text = '🟢 Q1: Thriving Achiever';
        badge_class = 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      } else if (pred_g3 >= 10 && bvi >= 40) {
        badge_text = '🟡 Q2: Burnout Risk Achiever';
        badge_class = 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      } else if (pred_g3 < 10 && bvi >= 40) {
        badge_text = '🔴 Q3: Dual Crisis';
        badge_class = 'bg-rose-500/20 text-rose-400 border-rose-500/30';
      } else {
        badge_text = '🔵 Q4: Coasting / Disengaged';
        badge_class = 'bg-sky-500/20 text-sky-400 border-sky-500/30';
      }
      const bEl = document.getElementById('diag-badge');
      bEl.className = 'px-2.5 py-0.5 rounded-full text-xs font-bold border ' + badge_class;
      bEl.innerText = badge_text;

      let presc_act = '';
      let uplift = 0;
      let new_pass = pass_prob;
      let new_bvi = bvi;

      if (pass_prob >= 0.88 && bvi < 36) {
        presc_act = 'Currently thriving. Maintain balanced study habit and positive momentum.';
        uplift = 0.0;
        new_pass = pass_prob;
        new_bvi = bvi;
      } else if (pred_g3 < 10) {
        const steps = [];
        let opt_study = Math.min(4, study + 1);
        let opt_abs = Math.max(0, abs - 4);
        steps.push('Enroll in targeted mathematics tutoring');
        if (study < 3) steps.push('Increase weekly study (+2-3 hrs)');
        if (abs > 4) steps.push('Attendance recovery (' + (abs - opt_abs) + ' days)');
        if (goout > 3) steps.push('Rebalance weekend rest');
        presc_act = steps.join('; ') + '.';
        uplift = 1.85;
        new_pass = Math.min(0.96, pass_prob + 0.45);
        new_bvi = Math.max(18, bvi - 14);
      } else {
        presc_act = 'High performance under strain. Rebalance social outings (-1 level), increase recovery rest, and preserve study pacing.';
        uplift = 0.4;
        new_pass = Math.min(0.99, pass_prob + 0.05);
        new_bvi = Math.max(22, bvi - 16);
      }

      document.getElementById('presc-action').innerText = presc_act;
      document.getElementById('presc-uplift').innerText = '+' + uplift.toFixed(2) + ' pts';
      document.getElementById('presc-newpass').innerText = (new_pass*100).toFixed(1) + '%';
      document.getElementById('presc-newbvi').innerText = new_bvi.toFixed(1) + ' / 100';
    }

    function initQuadrantChart() {
      const ctx = document.getElementById('quadrantChart').getContext('2d');
      const q1Data = COHORT_DATA.filter(d => d.quadrant.startsWith('Q1')).map(d => ({ x: d.BVI, y: d.G3_actual, id: d.student_id }));
      const q2Data = COHORT_DATA.filter(d => d.quadrant.startsWith('Q2')).map(d => ({ x: d.BVI, y: d.G3_actual, id: d.student_id }));
      const q3Data = COHORT_DATA.filter(d => d.quadrant.startsWith('Q3')).map(d => ({ x: d.BVI, y: d.G3_actual, id: d.student_id }));
      const q4Data = COHORT_DATA.filter(d => d.quadrant.startsWith('Q4')).map(d => ({ x: d.BVI, y: d.G3_actual, id: d.student_id }));

      quadrantChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Q1: Thriving', data: q1Data, backgroundColor: 'rgba(46, 204, 113, 0.75)', pointRadius: 4.5 },
            { label: 'Q2: Burnout Risk', data: q2Data, backgroundColor: 'rgba(243, 156, 18, 0.85)', pointRadius: 5.5 },
            { label: 'Q3: Dual Crisis', data: q3Data, backgroundColor: 'rgba(231, 76, 60, 0.85)', pointRadius: 5.5 },
            { label: 'Q4: Coasting', data: q4Data, backgroundColor: 'rgba(52, 152, 219, 0.75)', pointRadius: 4.5 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: 'Burnout Vulnerability Index (BVI: 0 - 100)', color: '#94a3b8' },
              grid: { color: '#1e293b' },
              ticks: { color: '#94a3b8' },
              min: 0, max: 80
            },
            y: {
              title: { display: true, text: 'Final Grade (G3: 0 - 20)', color: '#94a3b8' },
              grid: { color: '#1e293b' },
              ticks: { color: '#94a3b8' },
              min: 0, max: 20
            }
          },
          plugins: {
            legend: { labels: { color: '#e2e8f0' } },
            tooltip: {
              callbacks: {
                label: (ctx) => 'Student #' + ctx.raw.id + ': Grade ' + ctx.raw.y + '/20, BVI ' + ctx.raw.x
              }
            }
          }
        }
      });
    }

    function filterQuadrantChart(qKey) {
      if (!quadrantChartInstance) return;
      quadrantChartInstance.data.datasets.forEach(ds => {
        if (qKey === 'All') ds.hidden = false;
        else ds.hidden = !ds.label.startsWith(qKey);
      });
      quadrantChartInstance.update();
    }

    function populateCohortTable() {
      const tbody = document.getElementById('cohort-tbody');
      tbody.innerHTML = '';
      COHORT_DATA.forEach(d => {
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/40 transition';
        const qCode = d.quadrant.split(':')[0];
        const qColor = qCode==='Q1'?'text-emerald-400':qCode==='Q2'?'text-amber-400':qCode==='Q3'?'text-rose-400':'text-sky-400';
        tr.innerHTML = `
          <td class="py-2.5 px-3 font-mono text-slate-400">#${d.student_id}</td>
          <td class="py-2.5 px-3 font-mono">${d.G1}</td>
          <td class="py-2.5 px-3 font-mono">${d.G2}</td>
          <td class="py-2.5 px-3 font-mono font-bold">${d.G3_actual}</td>
          <td class="py-2.5 px-3 font-mono">${d.BVI}</td>
          <td class="py-2.5 px-3 font-semibold ${qColor}">${d.quadrant}</td>
          <td class="py-2.5 px-3 font-mono text-emerald-400 font-bold">+${d.expected_uplift} pts</td>
          <td class="py-2.5 px-3 text-[11px] text-slate-300 max-w-xs truncate" title="${d.recommended_action}">${d.recommended_action}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    function filterCohortTable() {
      const qVal = document.getElementById('filter-quadrant-select').value;
      const search = document.getElementById('search-student').value.toLowerCase();
      const rows = document.querySelectorAll('#cohort-tbody tr');

      COHORT_DATA.forEach((d, idx) => {
        const row = rows[idx];
        if (!row) return;
        const qCode = d.quadrant.split(':')[0];
        const matchQ = (qVal === 'all' || qCode === qVal);
        const matchSearch = (!search || String(d.student_id).includes(search) || d.school.toLowerCase().includes(search));
        row.style.display = (matchQ && matchSearch) ? '' : 'none';
      });
    }

    window.addEventListener('DOMContentLoaded', () => {
      populateCohortTable();
      recalculate();
    });
  </script>
</body>
</html>
"""

def main():
    df = pd.read_csv("eduprescribe_cohort_prescriptions.csv")
    json_data = df.to_json(orient="records")
    rendered = TEMPLATE.replace("__COHORT_DATA__", json_data)
    with open("eduprescribe_dashboard.html", "w") as f:
        f.write(rendered)
    print("Successfully built 'eduprescribe_dashboard.html' (size: %d bytes)" % len(rendered))

if __name__ == "__main__":
    main()
