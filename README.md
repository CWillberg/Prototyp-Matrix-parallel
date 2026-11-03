# Prototyp: Parallele Sparse-Matrix für Peridynamik

Prototyp zum Aufbau und Lösen einer dünnbesetzten Systemmatrix (Steifigkeitsmatrix)
für Peridynamik-Simulationen in Julia, verteilt über MPI und lauffähig auf CUDA-GPUs.

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
- **A3.6** Weitere Modelle (state-based, Korrespondenz) folgen später über dieselbe
  Schnittstelle, ohne die Assemblierung zu ändern.

### A4 – Aufbau (Assemblierung)
- **A4.0** Die Assemblierung kennt kein Materialmodell. Sie verteilt nur die Werte der
  Element-Matrizen `K_e` anhand von `row_dofs`/`col_dofs` in die globale Matrix.
- **A4.0a** Das Besetzungsmuster wird ebenfalls nur aus `row_dofs`/`col_dofs` aller Elemente
  ermittelt (symbolische Phase), nicht aus modellspezifischem Wissen.
- **A4.0b** Trägt ein Element in Zeilen ein, die einem anderen Rank gehören (Ghost-Zeilen),
  werden diese Beiträge gesammelt und zum Eigentümer-Rank gesendet und dort aufaddiert.
  Schreibt jedes Element nur in die Zeilen seines eigenen Punktes, entfällt diese
  Kommunikation (siehe F1).
- **A4.1** Die Assemblierung läuft parallel: verteilt über MPI-Ranks und innerhalb eines
  Ranks als GPU-Kernel.
- **A4.2** Partitionierung zeilenweise nach Punkt-Eigentümerschaft; jeder Rank besitzt seine
  Zeilen. Nachbarn auf anderen Ranks werden als Ghost-/Halo-Punkte geführt.
- **A4.3** Die Assemblierung auf der GPU ist ohne Race-Conditions (z. B. vorab berechnete
  Zielindizes je Bond statt atomarer Additionen – offen, siehe F4). Schreibt jedes Element
  nur in eigene Zeilen (bond-based), gibt es keine Konflikte: ein Thread pro Element genügt,
  Färbung (PeriLab-Konzept, Variante A) ist erst für Elemente nötig, die in fremde Zeilen
  schreiben.
- **A4.4** Keine Host↔Device-Kopien der Matrix während der Assemblierung.

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
- **A6.3** Die Gesamtmatrix bleibt unverändert, sodass sich Randbedingungen zwischen
  Lastschritten ändern können, ohne neu zu assemblieren.
- **A6.4** Vorgeschlagenes Verfahren: maskierter Operator
  - Anwendung: `y = M_f · K · (M_f · x) + M_p · x` (M = Diagonalmaske, nur als Bitvektor
    gespeichert, in den SpMV-Kernel integriert)
  - rechte Seite: `b = M_f · (f − K · (M_p · u_p))` + `M_p · u_p`
  - Damit ist nur ein Arbeitsvektor nötig, keine zusätzliche Matrix.
- **A6.5** Reaktionskräfte an den vorgegebenen Freiheitsgraden sind aus der vollen Matrix
  berechenbar (`r_p = M_p · (K u − f)`).

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

- **F1** Schreibt ein Element nur in die Zeilen seines eigenen Punktes `i`, oder auch in die
  Zeilen der Nachbarn (z. B. Bond-Kraft `t_ij` wirkt auf `i` und `j`)? Ersteres spart die
  Rückkommunikation der Ghost-Zeilen. *Für bond-based geklärt: nur eigene Zeilen (A3.5).
  Für state-based/Korrespondenz später erneut prüfen; die Assemblierung unterstützt
  beides (A4.0b).*
- **F2** ~~Format/Paket~~ – vorläufig geklärt: `PartitionedArrays.jl` (A8.3), Test A7.4 offen.
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
