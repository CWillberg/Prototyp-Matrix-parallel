# Prototyp: Parallele Sparse-Matrix für Peridynamik

Prototyp zum Aufbau und Lösen einer dünnbesetzten Systemmatrix (Steifigkeitsmatrix)
für Peridynamik-Simulationen in Julia, verteilt über MPI und lauffähig auf CUDA-GPUs.

## Stand

| Bereich | Stand | Ort |
|---|---|---|
| CSR-Matrix, SpMV (CPU/GPU) | fertig | `src/Core/Sparse_CSR.jl` |
| Assemblierung, modellunabhängig (symbolisch + Gather ohne Atomics) | fertig, auch für Elemente mit fremden Zeilen getestet | `src/Core/Matrix_Assembly.jl` |
| Element-Schnittstelle + Plugin-Suche | fertig | `src/Models/Elements/` |
| Bond-based Elastic (1D/2D/3D) | fertig | `src/Models/Elements/BondBased/` |
| DOF-Mengen, maskierte Operatoren, RB ohne Teilmatrizen | fertig | `src/Core/Dof_Sets.jl`, `src/Core/Operators.jl` |
| GMRES (Krylov.jl), Vorkonditionierer-Schnittstelle | fertig | `src/Core/Solver/` |
| Guyan: explizit (`extract`) und matrixfrei (`SchurOperator`) | fertig | `src/Core/Model_reduction/` |
| Korrespondenz Elastic (linear, mit Null-Energie-Stabilisierung, 1D–3D, GPU) | fertig | `src/Models/Elements/Correspondence/` |
| Ordinary, IGA-PD | offen (Schnittstelle und Verteilung vorbereitet) | – |
| Craig-Bampton | offen | – |
| Blockassemblierung (feste RB, A6.6) | offen | – |
| Status-Maske im Gather (Schädigung, A6a.2) | offen | – |
| Präfix-Sicht für additive Fertigung (A6a.3) | offen | – |
| Newton / Wiederverwendung / teilweises Neueinsammeln (A6b) | offen | – |
| weitere Vorkonditionierer, externe Löser (F7) | offen | – |
| `@timeit`-Abschnitte (Zeit + Allokation) | fertig | `test/benchmarks/profile_bar.jl` |
| MPI: verteilte Assemblierung (eigene Zeilen × eigene+Ghost-Spalten), Ghost-Austausch, verteilte Skalarprodukte | fertig (CPU- und GPU-Ranks; Elemente nur mit eigenen Zeilen) | `src/MPI_communication/`, `src/Core/Distributed_Operators.jl` |
| BiCGStab mit Jacobi (seriell / MPI / GPU, 0 B Allokation in der Iteration) | fertig | `src/Core/Solver/BiCGStab.jl` |
| Speicherbericht pro Rank | fertig | `src/Support/Memory_Report.jl` |
| Elemente, die in Nachbarzeilen schreiben, verteilt: 1-Horizont-Ghosts + Rückaddition im Löser (A4.0b) | fertig | `accumulate_ghosts!` in `src/MPI_communication/Halo_Exchange.jl` |

## Nutzung

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'        # inkl. GPU, wenn CUDA funktioniert
PD_MATRIX_TEST_GPU=false julia --project=. -e 'using Pkg; Pkg.test()'   # nur CPU
julia --project=. examples/bb_bar.jl                 # Referenzbeispiel A7.1a
julia -e 'using JuliaFormatter; format(".")'         # Formatierung (sciml)
julia --project=test/benchmarks -t 1 test/benchmarks/profile_bar.jl 1000       # Zeit/Allokation je Abschnitt
julia --project=test/benchmarks -t 4 test/benchmarks/profile_bar.jl 1000 gpu   # 4 Threads (WSL-Maximum) bzw. GPU
```

### Verteilte Testszenarien (MPI)

```sh
julia --project=test/mpi -e 'using Pkg; Pkg.instantiate()'          # einmal
julia --project=test/mpi test/mpi/run_scenarios.jl                   # CPU, 1/2/4 Ranks
julia --project=test/mpi test/mpi/run_scenarios.jl cpu gpu ranks=1,2  # zusätzlich GPU-Ranks
# einzeln (mpiexec aus MPI.jl: julia -e 'using MPI; println(mpiexec())'):
mpiexecjl -n 4 julia --project=test/mpi test/mpi/scenarios.jl cpu memory
mpiexecjl -n 2 julia --project=test/mpi test/mpi/scenarios.jl bc_values timer
```

| Szenario | Inhalt | Prüfungen |
|---|---|---|
| `halo` | Ghost-Austausch, 1D (D = 1) und 2D (D = 2) | Ghost-Werte = Werte der Eigentümer |
| `bar_uniform` | 1D-Stab, 100 Punkte, u = 0 / 100, F = 10 in der Mitte (A7.1a) | Zeilen von K = Zeilen der seriellen Matrix, u = Referenz, u_p exakt, Reaktionen, Gleichgewicht |
| `bar_nonsym` | wie oben, ungleiche Volumina | wie oben, Matrix nicht symmetrisch |
| `plate_2d` | 2D-Platte 24×8, links eingespannt, rechts u_x = 1 | wie `bar_uniform` |
| `bc_values` | neue RB-Werte und Last ohne Neuaufbau | Lösung = Referenz, Matrixwerte unverändert |
| `corr_bar` | 1D-Korrespondenz, 60 Punkte, ungleiche Volumina, Stabilisierung | Rückaddition aktiv, verteiltes K·x = serielles K·x, u, u_p exakt, Reaktionen |
| `corr_plate` | 2D-Korrespondenz 16×6, links eingespannt, rechts u_x = 0.5 | wie `corr_bar` |
| `memory` | Stab mit 10 000 Punkten, Speichertabelle pro Rank | Σ nnz lokal = nnz global, kein Rank hält die ganze Matrix |

Optionen: `cpu` / `gpu`, Szenarionamen (Standard: alle), `timer` (Zeit und Allokation je
Abschnitt auf Rank 0, nach einem Aufwärmlauf). Exit-Code 1 bei einem Fehler.
`Pkg.test()` startet die Szenarien zusätzlich mit 1 und 2 Ranks (abschaltbar mit
`PD_MATRIX_TEST_MPI=false`).

Auf einem Rechner mit einer GPU teilen sich mehrere GPU-Ranks dieselbe GPU: Das prüft die
Korrektheit, nicht die Leistung. Ohne CUDA-aware MPI läuft der Ghost-Austausch über
vorab angelegte Host-Puffer (F6).

### Studie: 1 CPU als Referenz, dann nur MPI, nur CUDA, gemischt

```sh
julia --project=test/scaling test/scaling/run_study.jl          # lokal
sbatch test/scaling/slurm_study.sh                              # Cluster
```

Pro Problemgröße: Referenz mit 1 CPU-Rank, dann CPU-Ranks (`cpu_ranks`), GPU-Ranks
(`gpu_ranks`, eine GPU pro Rank) und gemischt (`mixed_ranks`, ein Rank pro GPU auf der GPU,
der Rest auf der CPU). Ergebnis in `results/study/`: CSV, eine Tabelle in der Konsole
(Lösungszeit und Speedup gegenüber 1 CPU) und ein Plot (Spalten = Problemgröße, Zeilen =
Lösungszeit und Speedup, Linien = MPI / CUDA / gemischt, gestrichelt = Referenz).
Lokal: `dofs=1000,3000,10000 cpu_ranks=2,4 gpu_ranks=1 mixed_ranks=2,4`.

### Skalierungsstudie (ein Startskript)

```sh
julia --project=test/scaling -e 'using Pkg; Pkg.instantiate()'        # einmal
julia --project=test/scaling test/scaling/run_scaling.jl               # alles durchspielen
```

Spielt alle Kombinationen aus Backend (`cpu`, `gpu`, `mixed`), Ranks und DOFs durch, schreibt
`results/scaling/scaling_<zeit>.csv` und erzeugt daraus
`results/scaling/scaling_<zeit>_<dim>D_<element>.png` (Lösungszeit und Zeit pro Iteration über
der Zahl der Ranks, eine Linie pro Problemgröße, Spalten = Backend).
Standard lokal: Ranks 1, 2, 4; DOFs 1 000 / 3 000 / 10 000; 2D bond-based.

- Jeder Rank erzeugt nur seine eigenen Punkte und Ghosts (strukturiertes Gitter, Scheiben
  entlang x); es gibt keine globalen Daten und keine globale Referenz. Geprüft werden
  Konvergenz, wahres Residuum der freien DOFs und exakte Vorgabewerte.
- `mixed`: pro Knoten ein Rank je GPU auf der GPU, die übrigen Ranks auf der CPU, gekoppelt
  über denselben Ghost-Austausch; `gpu_weight` gibt GPU-Ranks mehr Punkte.
- GPU-Zuordnung: Gerät = Rank innerhalb des Knotens mod Anzahl GPUs.
- CSV-Spalten: Setup-, Assemblierungs-, Lösungszeit, Iterationen, Zeit pro Iteration,
  Kommunikationszeit im Löser, Speicher (max. Rank / Summe), Residuum.
- Optionen: `ranks=`, `dofs=`, `backends=`, `dims=`, `element=bb|corr`, `gpu_weight=`,
  `launcher=mpiexec|srun`, `threads=`, `horizon=`, `rtol=`, `itmax=`, `out=`.
- Cluster: `test/scaling/slurm_scaling.sh` (System-MPI über `MPIPreferences`, `launcher=srun`).

Ablauf seriell (siehe `test/helper.jl`, `solve_bar`):

1. Host, symbolisch: `element_dofs(el, data)` → `create_assembly_pattern` (einmal).
2. `adapt(CuArray, pattern)`, `adapt(CuArray, data)` für die GPU (oder `Array` für CPU).
3. Numerisch: `compute_elements!(ke, pattern, el, data)` → `assemble!(K, pattern, ke)`.
4. RB: `create_dof_set`, `constrained_rhs!`, `solve_linear!(ws, ConstrainedOperator(K, bc), b)`,
   `apply_prescribed!`, `reactions!`.

## Anforderungen

### A0 – Struktur, Stil und Architektur (Vorbild: PeriLab)
Der Prototyp orientiert sich an [PeriLab.jl](https://github.com/PeriHub/PeriLab.jl), damit er
später ohne Umbau übernommen werden kann.

- **A0.0 Eigenständig, später PeriLab-Modul**
  - Der Prototyp ist zunächst ein **eigenständiges Julia-Paket ohne Abhängigkeit zu PeriLab**
    (kein `using PeriLab`, kein Import von PeriLab-Modulen).
  - Er wird so gebaut, dass er später als Modul in PeriLab eingehängt werden kann
    (z. B. unter `src/Core/Solver/` bzw. `src/Core/Matrix/`), ohne seinen Code zu ändern:
    - Modulname und Unterbau nach PeriLab-Konventionen (A0.1/A0.2),
    - einzige Schnittstelle sind Funktionsargumente (A0.3), keine globalen Zustände,
    - relative Importe nur innerhalb des Prototyps; beim Einhängen ändert sich nur die
      Modulwurzel.
  - Abhängigkeiten: **freie Paketwahl**. Die Wahl richtet sich nach Eignung (MPI, GPU,
    nicht-symmetrische Matrizen), nicht danach, was PeriLab bereits nutzt.
    Empfehlung: `CUDA.jl` als Paket-Erweiterung (weak dependency / `ext/`) einbinden, damit
    der Code auch ohne GPU-Pakete lauffähig bleibt.

- **A0.1 Stil**
  - SciML Style Guide, erzwungen per JuliaFormatter (`.JuliaFormatter.toml` aus PeriLab
    übernehmen: `style = "sciml"`), pre-commit-Hooks (Formatter, REUSE, Whitespace).
  - `snake_case` für Funktionen/Variablen, sprechende Namen; Module
    `Capitalized_With_Underscores` (z. B. `Matrix_Assembly`).
  - Typannotationen an Funktionssignaturen, `Int64`/`Float64` als Standardtypen.
  - Höchstens 4 Verschachtelungsebenen; sonst Funktion auslagern oder früh `return`/`continue`.
  - Docstrings im PeriLab-Format (Signatur, `# Arguments`, `# Returns`, Beispiel).
  - SPDX-Lizenzkopf in jeder Datei (BSD-3-Clause).
  - Eingaben nur bei der Initialisierung prüfen, nicht in der Rechenschleife.
- **A0.2 Struktur**
  - `src/` als Modulbaum wie in PeriLab; Unit-Tests unter `test/unit_tests/**/ut_*.jl`
    spiegeln `src/` 1:1; Gesamttests unter `test/fullscale_tests/test_<Name>/`.
  - Elemente (Materialmodelle) nach dem PeriLab-Plugin-Muster: eine Datei/ein Modul pro
    Element mit gleich benannten Schnittstellenfunktionen (z. B. `element_name()`,
    `init_element!`, `compute_element!`). Ein neues Element = eine neue Datei.
    Da der Prototyp eigenständig ist, wird PeriLabs `@params` / `register_material` nicht
    verwendet; die Parameter kommen als einfacher (GPU-tauglicher, `isbits`) Struct.
    Beim Einhängen in PeriLab verbindet ein dünner Adapter beide Welten.
  - Module gleicher Ebene rufen sich nicht direkt auf (z. B. Element ↔ Löser), sondern
    gehen über die gemeinsame Schnittstelle.
  - Vorgeschlagener Baum:
    ```
    src/
      Prototyp.jl                 # Modulwurzel
      Core/
        Matrix_Assembly/          # Muster (symbolisch) + Werte verteilen (numerisch)
        Solver/                   # Krylov-Löser, maskierter Operator
        BC_manager.jl             # Randbedingungen als Maske (A6)
      Models/
        Elements/
          Element_template/       # Vorlage
          BondBased/              # erstes Element (A3.5)
      Backend/                    # CPU/CUDA-Auswahl (KernelAbstractions)
    test/
      unit_tests/ …               # spiegelt src/
      fullscale_tests/ …
    ```
- **A0.3 Daten werden von außen geliefert**
  - Die Module des Prototyps haben **keinen Zugriff auf `Data_Manager`** und bauen auch
    keinen eigenen auf. Alle Eingaben kommen als Funktionsargumente.
  - Der Aufrufer (PeriLab) holt die Daten aus `Data_Manager` und übergibt sie. Dort liegen
    bereits alle Daten der eigenen Punkte **und ihrer Nachbarn (Ghosts)** vor; der
    Halo-Austausch der Eingangsfelder ist Sache des Aufrufers, nicht des Prototyps.
  - Gleiche Signatur auf CPU und GPU: Übergeben werden `AbstractArray`s; ob es `Array` oder
    `CuArray` ist, entscheidet der Aufrufer. Die Funktionen allozieren nichts Neues in der
    Rechenschleife und schreiben in übergebene Puffer (`!`-Funktionen).

  | Vom Aufrufer geliefert | PeriLab-Quelle (nur zur Orientierung) |
  |---|---|
  | Koordinaten, Volumen, Horizont, Nachbarliste, Bond-Schaden | `Data_Manager.get_field`, `get_nlist`, `get_bond_damage` |
  | DOF pro Punkt, Anzahl eigener/Ghost-Punkte | `get_dof`, `get_nnodes` |
  | Bond-based-Konstante | `Material_Basis.compute_bond_based_constants` |
  | MPI-Kommunikator, Owner/Ghost-Zuordnung (`distribution`, `glob_to_loc`, `overlap_map`) | `get_comm`, `get_rank`, Zerlegung |
  | Maske der vorgegebenen DOFs + Werte `u_p` | `BC_manager` (`find_bc_free_dof`, `apply_bc_dirichlet`) |

  Der Prototyp liefert zurück: Matrix (Muster + Werte), Lösung `u`, ggf. Reaktionskräfte.
- **A0.3a** Kommunikation, die der Prototyp selbst braucht (Austausch des Lösungsvektors
  bei SpMV im Löser, Fremd-Zeilen bei der Assemblierung, Skalarprodukte), läuft über den
  übergebenen Kommunikator und die übergebene Owner/Ghost-Zuordnung. Dafür werden
  `MPI.jl` bzw. `PartitionedArrays.jl` genutzt. Die Owner/Ghost-Zuordnung wird im selben
  Format übernommen, das PeriLab `MPI_Communication` verwendet (`overlap_map`), damit beim
  Einhängen keine Umrechnung nötig ist.
- **A0.3b** Tests erzeugen die Eingabedaten selbst (kleines Gitter + Nachbarliste in
  `test/helper.jl`), ohne PeriLab laufen zu lassen. Der serielle PeriLab-Matrixpfad
  (`Correspondence_matrix_based`, `Matrix_linear_static`) dient als Vergleich.
- **A0.4** Das Konzept in PeriLab `PARALLEL_STIFFNESS_MATRIX.md` ist die Grundlage
  (Owner-Zeilen pro Rank, rank-sortierte globale DOF-Nummerierung, PartitionedArrays/PETSc).
  Der Prototyp setzt dessen Stufe 2 (MPI) um und ergänzt CUDA.

### A1 – Sprache und Umgebung
- **A1.1** Implementierung in Julia (Version ≥ 1.10; CI wie PeriLab auf 1.12).
- **A1.2** Der Code ist backend-agnostisch: dieselbe Implementierung läuft auf CPU und GPU (CUDA).
- **A1.3** Wo möglich werden bestehende Module genutzt statt Eigenentwicklungen, in dieser
  Reihenfolge:
  1. Muster und Algorithmen aus PeriLab übernehmen (Konventionen, Konzept
     `PARALLEL_STIFFNESS_MATRIX.md`), aber **nicht importieren** (A0.0),
  2. etablierte Julia-Pakete: `MPI.jl`, `CUDA.jl`, `KernelAbstractions.jl`,
     `PartitionedArrays.jl`, `Krylov.jl`, `LinearSolve.jl`, ggf. `PETSc.jl` / `HYPRE.jl`
     (Auswahl wie in PeriLab `PARALLEL_STIFFNESS_MATRIX.md`),
  3. eigener Code nur dort, wo nichts passt (z. B. GPU-Kernel für Assemblierung und
     maskierten Operator).

### A2 – Matrixstruktur
- **A2.1** Die Matrix ist dünnbesetzt (sparse). Das Besetzungsmuster ergibt sich aus der
  Nachbarschaftsliste (Bonds innerhalb des Horizonts δ).
- **A2.2** Die Matrix ist im Allgemeinen **nicht symmetrisch** (z. B. Korrespondenzmodelle,
  variabler Horizont, ungleichmäßige Diskretisierung, Schädigung). Symmetrie darf weder
  beim Aufbau noch beim Lösen vorausgesetzt werden.
- **A2.3** Freiheitsgrade: `dof = dim × n_punkte` (dim = 1, 2 oder 3), blockweise Anordnung pro Punkt.
- **A2.4** Speicherformat: **CSR**. Die Struktur bleibt verlustfrei in BSR (dim×dim-Blöcke)
  umwandelbar, BSR kommt später als GPU-Optimierung:
  - DOFs knotenweise nummeriert (A2.3), sodass jeder Block zusammenhängt,
  - Besetzungsmuster aus dem Punkt-Graphen abgeleitet: jedes Punktpaar `(i, j)` erzeugt
    immer den vollen dim×dim-Block (auch explizite Nullen),
  - damit ist BSR = CSR mit gleichem Muster auf Punktebene; für dim = 1 sind beide identisch.
- **A2.5** Besetzungsmuster und Werte sind getrennt: Das Muster wird einmal erstellt,
  die Werte können (z. B. nach Schädigung / pro Lastschritt) neu assembliert werden,
  ohne das Muster neu zu allozieren.

### A3 – Elemente (modellunabhängige Schnittstelle)
- **A3.0** Die Matrix ist unabhängig vom Materialmodell. Das Modell (bond-based, state-based,
  Korrespondenz, …) liefert nur „Elemente“.
- **A3.1** Ein **Element** ist die Integration an einem Punkt `i`. Es liefert:
  - `row_dofs` – Freiheitsgrade, in deren Zeilen das Element einträgt
    (typisch: die von Punkt `i`),
  - `col_dofs` – Freiheitsgrade, von denen das Element abhängt
    (Punkt `i` und seine Nachbarn),
  - `K_e` – dichte, im Allgemeinen rechteckige und nicht-symmetrische lokale Matrix
    der Größe `|row_dofs| × |col_dofs|`.
- **A3.2** Die Elemente werden unabhängig voneinander berechnet (pro Kern/Rank bzw.
  pro GPU-Thread). Zwischen den Elementen gibt es keine Abhängigkeiten.
- **A3.3** Jeder Rank berechnet nur die Elemente seiner eigenen Punkte. Für Nachbarn auf
  anderen Ranks liegen deren Daten bereits als Ghost-Punkte vor; sie werden vom Aufrufer
  geliefert (A0.3).
- **A3.4** Die Elementschnittstelle ist auf der GPU aufrufbar (keine Allokationen, keine
  dynamischen Typen im Kernel); `K_e` wird in einen vorallozierten Puffer geschrieben.
- **A3.5** Erstes Element: **bond-based** (einfach, gut zu debuggen). Linearisiert gilt für
  Punkt `i` mit Nachbarn `j ∈ H_i`:
  - `k_ij = c · ω_ij · V_j / |ξ_ij| · (e_ij ⊗ e_ij)` mit `ξ_ij = x_j − x_i`, `e_ij = ξ_ij / |ξ_ij|`
  - Beiträge: `K[i,i] += k_ij`, `K[i,j] −= k_ij` (jeweils dim×dim-Blöcke)
  - `row_dofs` = DOFs von `i`, `col_dofs` = DOFs von `i` und allen `j ∈ H_i`.
  - Das Element schreibt nur in die Zeilen des eigenen Punktes → keine Rückkommunikation
    von Ghost-Zeilen nötig.
  - Bei gleichmäßiger Diskretisierung ist das Ergebnis symmetrisch; durch unterschiedliche
    Volumina `V_j`, Horizonte oder Schädigung `ω_ij` wird es nicht-symmetrisch. Das eignet
    sich gut als Test für A2.2.
- **A3.6** Die Basis muss von Anfang an folgende Elementtypen tragen, ohne die
  Assemblierung zu ändern (implementiert wird zunächst nur A3.5):

  | Elementtyp | Integrationspunkt | `row_dofs` | `col_dofs` | schreibt in Nachbarzeilen | Zeile der zusammengesetzten Matrix reicht |
  |---|---|---|---|---|---|
  | bond-based (A3.5) | Materialpunkt `i` | `i` | `{i} ∪ H_i` | nein | 1 Horizont |
  | ordinary state-based | Materialpunkt `i` | `{i} ∪ H_i` (Kraftzustand `T_i⟨ξ⟩` wirkt auf `i` und `j`) | `{i} ∪ H_i` (Dilatation `θ_i`) | ja | 2 Horizonte |
  | Korrespondenz | Materialpunkt `i` | `{i} ∪ H_i` | `{i} ∪ H_i` (Formtensor, Deformationsgradient) | ja | 2 Horizonte |
  | IGA-PD | Gauß-Punkt `x` der äußeren Integration | Kontrollpunkte mit `N_I(x) ≠ 0` | Kontrollpunkte aller Elemente im Horizont `[x−δ, x+δ]` | ja | Träger + Horizont |

  Warum 2 Horizonte: Das Element am Punkt `i` hängt nur von seinen Nachbarn ab (Spannung
  aus Formtensor und Deformationsgradient an `i`). Die Bondkraft wirkt aber auf beide
  Enden (`+T_i⟨ξ_ij⟩` auf `i`, `−T_i⟨ξ_ij⟩` auf `j`). Zeile `i` sammelt deshalb auch die
  Elemente der Nachbarn `j`, deren Spalten `N(j)` sind. Beispiel 1D, Horizont 1·Δx:
  `f_4 = (σ_5 − σ_3)/2 = E/4 (u_2 − 2u_4 + u_6)`, also `K_42 ≠ 0` ohne Bond 4–2
  (Test `ut_Correspondence_Elastic`).

- **A3.7** Folgen für die Schnittstelle:
  - Freiheitsgrade gehören zu **DOF-Trägern** (Materialpunkte *oder* Kontrollpunkte). Ein
    Element ist an einen Integrationspunkt gebunden, nicht zwingend an einen DOF-Träger.
  - `row_dofs` und `col_dofs` sind beliebige Listen; `K_e` ist dicht und rechteckig
    (IGA-PD: `K_e = w_x · N(x) · g(x)ᵀ`, siehe IGA_PD `bar_plane/src/IGAPD.jl`).
  - Verteilt: siehe A4.0b (1-Horizont-Ghosts, Teilsummen der Ghost-Zeilen werden im Löser
    zum Eigentümer addiert).
  - Ein Element meldet seine DOF-Listen in einer symbolischen Phase einmal an
    (`element_dofs`); die numerische Phase liefert nur noch Werte in fester Reihenfolge.
  - Referenz für IGA-PD: Projekt `IGA_PD` (`~/julia_projects/IGA_PD`), 1D-Stab und 2D in
    `bar_plane/`, Schale in `shell/`; dort bisher dichte Matrizen und seriell.

### A4 – Aufbau (Assemblierung)
- **A4.0** Die Assemblierung kennt kein Materialmodell. Sie verteilt nur die Werte der
  Element-Matrizen `K_e` anhand von `row_dofs`/`col_dofs` in die globale Matrix.
- **A4.0a** Das Besetzungsmuster wird ebenfalls nur aus `row_dofs`/`col_dofs` aller Elemente
  ermittelt (symbolische Phase), nicht aus modellspezifischem Wissen.
- **A4.0b** Verteilung mit **einem Horizont Ghosts** (wie PeriLab), Entscheidung:
  - Jeder Rank rechnet nur die Elemente seiner eigenen Punkte. Die lokale Matrix hat die
    eigenen Zeilen **und** die Ghost-Zeilen; in den Ghost-Zeilen stehen Teilsummen
    (Beiträge der eigenen Elemente zu den Nachbarzeilen). Keine doppelte Berechnung, keine
    Kommunikation beim Aufbau, keine zweite Ghost-Schicht.
  - Die gesamte Zeile existiert auf keinem Rank. Sie entsteht im Löser:
    1. `exchange!` – Ghost-Werte von `x` von den Eigentümern (1 Horizont),
    2. lokale SpMV über alle lokalen Zeilen,
    3. `accumulate_ghosts!` – Ghost-Zeilen von `y` zum Eigentümer schicken und addieren
       (Muster von PeriLab `synch_responder_to_controller`, `add = true`).
  - Ebenso: rechte Seite, Jacobi-Diagonale, Reaktionskräfte.
  - Bond-based: Ghost-Zeilen leer, Schritt 3 entfällt automatisch (`needs_accumulate`).
  - Preis: zwei Austausche pro SpMV statt einem, dafür kleine Vektoren (1 Horizont) und
    weniger Speicher als 2-Horizont-Ghosts (3D, 20³ Punkte pro Rank, δ = 3Δx: 9 576
    statt 24 768 Ghost-Punkte).
  - Externe Löser, die fertige Zeilen erwarten (PETSc AIJ, ILU), brauchen einen
    zusätzlichen Schritt; PETSc MATIS nimmt diese nicht zusammengesetzte Form direkt.
- **A4.1** Die Assemblierung läuft parallel: verteilt über MPI-Ranks und innerhalb eines
  Ranks als GPU-Kernel.
- **A4.2** Partitionierung zeilenweise nach Punkt-Eigentümerschaft; jeder Rank besitzt seine
  Zeilen. Nachbarn auf anderen Ranks werden als Ghost-/Halo-Punkte geführt.
- **A4.3** Die Assemblierung auf der GPU ist ohne Race-Conditions (z. B. vorab berechnete
  Zielindizes je Bond statt atomarer Additionen – offen, siehe F4). Schreibt jedes Element
  nur in eigene Zeilen (bond-based), gibt es keine Konflikte: ein Thread pro Element genügt.
  Umgesetzt: Gather über vorberechnete Zuordnung (ein Work-Item pro Matrixeintrag), damit
  auch für Elemente, die in Nachbarzeilen schreiben, ohne Färbung und ohne Atomics.
- **A4.4** Keine Host↔Device-Kopien der Matrix während der Assemblierung.

### A4a – Modellreduktion
- **A4a.1** Die Basis muss Modellreduktion erlauben: statische Kondensation (Guyan) und
  Craig-Bampton wie in PeriLab `src/Core/Model_reduction/` (`reduce_matrices(K, M, m, s, n_modes)`).
- **A4a.2** Master- (`m`) und kondensierte (`s`) DOFs werden wie die Randbedingungen als
  **DOF-Mengen** (Maske) beschrieben. Eine gemeinsame Abstraktion für RB und Reduktion.
- **A4a.3** Operatoren auf Teilmengen ohne Kopie: `K_ab · x` für beliebige Mengen `a, b`
  als maskiertes SpMV auf der Gesamtmatrix (wie A6.4). Damit lassen sich `T = −K_ss⁻¹ K_sm`
  und `K_r = K_mm + K_ms T` iterativ anwenden, ohne Teilmatrizen zu bilden.
- **A4a.4** Wo ein Verfahren eine explizite Teilmatrix braucht (Faktorisierung von `K_ss`,
  Eigenwertproblem bei Craig-Bampton), gibt es eine **ausdrückliche** Extraktion
  `extract(K, a, b)`. Sie ist der einzige Ort, an dem Teilmatrizen alloziert werden, und wird
  nie implizit (z. B. durch Randbedingungen) aufgerufen.
- **A4a.5** Optionale Umnummerierung, sodass die DOFs einer Menge zusammenhängen
  (PeriLab `create_permutation`); die Zuordnung alt ↔ neu wird gespeichert.

### A5 – Lösung
- **A5.1** Lösung des linearen Systems `K u = f` für nicht-symmetrische Matrizen mit
  iterativen Krylov-Verfahren (GMRES, BiCGStab).
- **A5.2** Der Löser läuft verteilt (MPI) und auf der GPU; SpMV und Skalarprodukte
  laufen auf dem Device, Reduktionen über MPI.
- **A5.3** Ein Vorkonditionierer ist einbindbar (Schnittstelle vorsehen; Auswahl zurückgestellt, F7).

### A6 – Randbedingungen (vorgegebene Verschiebungen u)
- **A6.1** Dirichlet-Randbedingungen (vorgegebene Verschiebungen `u_p`) werden
  **ohne zusätzlichen Speicher für Untermatrizen** umgesetzt. Es werden keine Kopien von
  `K_ff`, `K_fp` o. Ä. angelegt.
- **A6.2** Die Aufteilung in freie (f) und vorgegebene (p) Freiheitsgrade erfolgt über
  eine Maske bzw. Indexliste (Speicher O(n_dof) als Bitmaske, nicht O(nnz)).
- **A6.3** Die **Werte** der Randbedingungen können sich jederzeit ändern, ohne neu zu
  assemblieren. Ändert sich die **Menge** der vorgegebenen DOFs, gilt: mit dem maskierten
  Operator (A6.4) ohne Neuaufbau, mit Blockassemblierung (A6.6) neue symbolische Phase.
- **A6.4** Verfahren für wechselnde RB-Mengen: maskierter Operator
  - Anwendung: `y = M_f · K · (M_f · x) + M_p · x` (M = Diagonalmaske, nur als Bitvektor
    gespeichert, in den SpMV-Kernel integriert)
  - rechte Seite (homogene Form): `b = M_f · (f − K · (M_p · u_p))`, also `b_p = 0`
  - Danach `u_p` exakt einsetzen (`apply_prescribed!`). Da `b_p = 0` und der Operator auf
    `p` die Identität ist, bleiben alle Krylov-Vektoren dort exakt 0; die vorgegebenen
    Werte sind damit ohne Rundungsfehler erfüllt.
  - Damit ist kein Arbeitsvektor und keine zusätzliche Matrix nötig.
- **A6.5** Reaktionskräfte an den vorgegebenen Freiheitsgraden sind aus der vollen Matrix
  berechenbar (`r_p = M_p · (K u − f)`).
- **A6.6** Für **feste** Randbedingungen (Menge `p` über die Rechnung konstant):
  **Blockassemblierung**. Die symbolische Phase verteilt die Elementwerte direkt in die
  Blöcke `K_ff`, `K_fp`, `K_pf`, `K_pp` mit neu nummerierten Indizes.
  - Zusammen genauso viele Einträge wie `K`, keine Kopie, kein Ausschneiden.
  - `K_ff` ist eine echte Matrix → jeder Löser und Vorkonditionierer ist nutzbar
    (Direktlöser, ILU, AMG, PETSc/HYPRE).
  - rechte Seite `f_f − K_fp u_p`, Reaktionen `r_p = K_pf u_f + K_pp u_p − f_p`.
  - Ändert sich die **Menge** `p`, muss die symbolische Phase neu laufen.
  - Allgemein: Jeder DOF bekommt ein Label (vorgegeben / Master / kondensiert), daraus
    bis zu 3×3 Blöcke; damit liefert dieselbe Assemblierung auch `K_mm`, `K_ms`, `K_sm`,
    `K_ss` für die Reduktion (A4a) ohne `extract`.
- **A6.7** Der maskierte Operator (A6.4) bleibt als Variante für Randbedingungen, deren
  Menge sich häufig ändert.

### A6a – Veränderliche Zustände: Schädigung und additive Fertigung
- **A6a.1 Schädigung** ändert nur Werte (`ω_ij = 0` in der Elementmatrix), nie das
  Besetzungsmuster.
- **A6a.2 Status-Maske im Gather-Kernel:** DOFs, die zur Laufzeit aus dem System fallen
  (Punkt hat alle Bonds verloren → Nullzeile), werden über eine Maske (O(n)) beim
  Einsammeln behandelt: Zeile und Spalte 0, Diagonale 1. Die Diagonale ist immer im
  Muster. Die Matrix bleibt eine echte Matrix (alle Löser nutzbar), das Muster bleibt,
  keine Kopie (wie PETSc `MatZeroRowsColumns`, aber direkt beim Aufbau).
- **A6a.3 Additive Fertigung – wachsende Matrix:**
  - DOFs werden **nach Aktivierungszeitpunkt** nummeriert (bekannt aus Baubahn / G-Code),
    sodass die aktiven DOFs immer `1:n_active` sind.
  - Das Muster wird einmal für die Endgeometrie gebaut, Spalten pro Zeile sortiert.
  - Die aktive Matrix ist der linke obere Block `K[1:n_active, 1:n_active]`, aber **nicht**
    als Indexierung (bei `SparseMatrixCSC` wäre das eine Kopie), sondern als
    **Präfix-Sicht** auf dieselben CSR-Arrays: ein Struct mit `rowptr`, `colval`, `nzval`
    und `n_active` (O(1) Speicher). Die Kernel laufen über die Zeilen `1:n_active` und
    brechen pro Zeile ab, sobald `colval[k] > n_active`.
  - Wachsen = `n_active` erhöhen. Assemblierung, SpMV, Löser und Vektoren (Views auf
    `1:n_active`) arbeiten nur auf dem aktiven Teil. Bonds zu inaktiven Punkten fallen
    automatisch weg.
  - Grenzen: Speicher für die Endgeometrie wird von Anfang an reserviert, Rechenzeit fällt
    nur für den aktiven Teil an. Externe Direktlöser (UMFPACK, Pardiso) brauchen eine
    eigenständige Matrix und damit eine Kopie des Blocks; Krylov-Löser und eigene
    Vorkonditionierer (Jacobi, ILU(0)) arbeiten direkt auf der Präfix-Sicht. Ist die
    Endgeometrie für den Speicher zu groß: echtes Wachstum mit erneuter symbolischer
    Phase alle paar Schichten.
  - Kombinierbar mit A6.6 (feste RB als Blöcke) und A6a.2 (Status-Maske).

### A6b – Nichtlineare Probleme
- **A6b.1** Newton-Verfahren mit wählbarer Strategie: volles Newton (Neuaufbau und neue
  Faktorisierung bzw. neuer Vorkonditionierer pro Iteration), modifiziertes Newton (einmal
  pro Lastschritt), Anfangssteifigkeit. Nie `K⁻¹` bilden, immer lösen.
- **A6b.2** Die numerische Assemblierung muss schnell sein (Neuaufbau pro Iteration);
  Muster und symbolische Analyse werden wiederverwendet.
- **A6b.3** Wiederverwendung über Iterationen/Lastschritte: symbolische Faktorisierung
  (Direktlöser: nur numerisch neu faktorisieren), Vorkonditionierer (neu nur bei
  steigender Iterationszahl), letzte Lösung als Startwert.
- **A6b.4** Teilweises Neueinsammeln: nur geänderte Elemente neu berechnen und nur die
  Matrixeinträge neu einsammeln, die sie berühren (Zuordnung aus dem Muster). Ergebnis
  bitgleich zum vollständigen Neuaufbau, keine Rundungsdrift (im Unterschied zu
  „alt abziehen, neu addieren“).
- **A6b.5** Eine lokale Aktualisierung der LU-Faktoren gibt es für nicht-symmetrische
  Matrizen praktisch nicht. Hebel bei lokaler Nichtlinearität: den linearen Bereich
  einmal kondensieren (Schur-Komplement wie A4a) und pro Iteration nur den nichtlinearen
  Rest plus Schnittstelle neu faktorisieren. Woodbury / Rang-k-Update nur bei sehr wenigen
  geänderten DOFs.

### A7 – Verifikation und Performance
- **A7.1** Testfälle: kleines Problem mit bekannter Lösung, Vergleich gegen serielle
  CPU-Referenz (`SparseArrays` + Direktlöser `\`).
- **A7.1a Referenzbeispiel: 1D-PD-Stab**
  - 100 Punkte, dim = 1 → Matrix 100 × 100, Element bond-based (A3.5).
  - Werte frei wählbar (z. B. Δx = 1, Horizont δ = 3·Δx, `c = 1`, `V = 1`).
  - Randbedingungen:
    - links (Punkt 1): `u = 0`,
    - rechts (Punkt 100): `u = 100 mm` (inhomogene Dirichlet-RB → testet die Korrektur der
      rechten Seite in A6.4),
    - Mitte (Punkt 50): Einzelkraft `F`.
  - Prüfungen:
    - Ergebnis gleich der seriellen Referenz,
    - gleiches Ergebnis auf 1, 2, 4 Ranks und CPU/GPU (A7.2),
    - kein zusätzlicher Speicher für Teilmatrizen (A6.1), z. B. per `@allocated` im Löseschritt,
    - Reaktionskräfte links/rechts im Gleichgewicht mit `F` (A6.5).
  - Variante mit ungleichmäßigen Volumina `V_j` → nicht-symmetrische Matrix (A2.2).
- **A7.2** Ergebnisse sind unabhängig von Rank-Anzahl und Backend (bis auf Rundungsfehler).
- **A7.3** Skalierungstest: starke und schwache Skalierung über Ranks/GPUs.
- **A7.4** Machbarkeitstests zu A8 (vor der eigentlichen Implementierung):
  - `JaggedArray(flatview(v), v.elem_ptr)` aus einem `VectorOfVectors` ohne Kopie
    (gleicher Speicher, `===` auf den Datenvektor).
  - GPU-Unterstützung der aktuellen `PartitionedArrays.jl`-Version: `PVector` mit
    `CuArray` als lokalen Werten, Ghost-Austausch (`consistent!`), Assemblierung (`assemble!`).
  - Ergebnis entscheidet, ob A8.4 Variante a oder b gilt.

### A8 – Datenstrukturen (vorläufig, Tests nach A7.4 stehen aus)
- **A8.1** Element-Eingaben (Nachbarliste, Bond-Felder, Bond-Schaden) bleiben
  `ArraysOfArrays` (`VectorOfVectors`, Typen wie PeriLab `BondField`). Sie werden nicht in
  PartitionedArrays-Strukturen überführt.
- **A8.2** Zielindizes für die Assemblierung: je Bond die Position in `nzval`, gespeichert als
  `VectorOfVectors` mit **demselben `elem_ptr` wie die Nachbarliste**. Einmal in der
  symbolischen Phase berechnet; der Kernel schreibt danach direkt
  `nzval[idx[i][j]] += …` (keine Suche, keine Atomics, keine Allokation). → F4
- **A8.3** Matrix als `PSparseMatrix` (lokal CSR), `u`/`f` als `PVector` aus
  `PartitionedArrays.jl`. → F2
- **A8.4** GPU:
  - a) PartitionedArrays unterstützt `CuArray` lokal → direkt nutzen.
  - b) sonst: PartitionedArrays nur für Index-Partition und Kommunikationsmuster; lokale
    Werte (`nzval`, Vektoren), SpMV und Kernel als flache Arrays mit KernelAbstractions.
- **A8.5** DOF-Reihenfolge im Prototyp knotenweise (`dof × nnodes`, A2.3). PeriLab liefert
  `NodeVectorField` als `nnodes × dof` (komponentenweise); an der Schnittstelle wird einmal
  umsortiert (O(n)).

## Offene Fragen

- **F1** ~~Eigene Zeilen oder Nachbarzeilen?~~ – geklärt: bond-based nur eigene Zeilen;
  Korrespondenz/Ordinary/IGA-PD auch Nachbarzeilen, verteilt über 1-Horizont-Ghosts mit
  Rückaddition im Löser (A4.0b).
- **F2** ~~Format/Paket~~ – geklärt: eigene CSR-Matrix + eigener Ghost-Austausch mit `MPI.jl`
  (klein, GPU-fähig, Format wie PeriLab `overlap_map`); `PartitionedArrays.jl` wird nicht
  verwendet. A8.3/A8.4 sind damit überholt.
- **F3** ~~CSR/BSR~~ – geklärt: CSR, in BSR umwandelbar gehalten (A2.4).
- **F4** ~~GPU-Assemblierung~~ – vorläufig geklärt: vorberechnete Zielindizes als
  `VectorOfVectors` (A8.2).
- **F5** ~~Partitionierung~~ – geklärt: kommt vom Aufrufer (PeriLab-Zerlegung, A0.3).
- **F6** Multi-GPU: CUDA-aware MPI voraussetzen? *Zurückgestellt.* Beim Entwurf
  berücksichtigen: Kommunikationspuffer austauschbar halten (Device direkt oder über Host).
- **F7** Welche Vorkonditionierer sind für die Zielprobleme nötig (Jacobi reicht? AMG?)?
  *Zurückgestellt.*
- **F8** ~~GPU und `Data_Manager`~~ – geklärt: Daten werden von außen geliefert (A0.3);
  der Aufrufer entscheidet, ob er `Array` oder `CuArray` übergibt.
- **F9** ~~PeriLab-Abhängigkeit~~ – geklärt: zunächst eigenständig, später als Modul in
  PeriLab integriert (A0.0).
