window.ADE_CODE_NARRATIVE = {
  "C1/P1": [
    "Complete targets",
    "Preserve complete supervision.",
    "Are cutoff-exceeding responses an avoidable source of supervision loss?",
    "Replace truncating baseline records with non-truncating candidates of similar length, keeping the 384-record budget and length allocation.",
    "Both validation metrics improved. The complete-target idea later crossed into another Coordinator’s combined strategy."
  ],
  "C1/P2": [
    "Judge validity",
    "Test semantic validity as a selection signal.",
    "Can a Judge identify invalid targets that should be replaced?",
    "Build on complete targets and seek same-family replacements using a semantic-validity rubric.",
    "The gate produced no failing group. Pass@3 rose slightly, but Average@3 fell; this informed the move to a deterministic signal."
  ],
  "C1/P3": [
    "Stratified repetition control",
    "Keep the exposure. Change the repetition.",
    "Does reducing repetition help when reasoning length and prompt type remain comparable?",
    "Replace highly repetitive parent records with low-repetition candidates matched by prompt type, truncation status and length band, then nearest token count.",
    "The best in-loop score, with the parent’s 142 / 107 / 135 length allocation preserved. Its hypothesis remained inconclusive; all three lanes reused the strategy."
  ],
  "C1/P4": [
    "Wrapper closure",
    "Check the whole executable wrapper.",
    "Is defining a requested function enough if the answer never calls it?",
    "Revisit C1/P3 and replace definite definition-and-call mismatches with matched wrapper-closed targets.",
    "The score fell below the parent. Inspection also exposed literal image markers, supplying a different question for C1/P5."
  ],
  "C1/P5": [
    "Image-marker removal",
    "Return to the stronger parent with a new clue.",
    "Do literal image placeholders weaken otherwise useful text-only supervision?",
    "Return to C1/P3 and replace marker-bearing records with matched marker-free candidates, retaining complete trajectories.",
    "The result did not exceed the leader. The diagnostic source and the inherited artifact came from different Plans."
  ],
  "C2/P1": [
    "Code-contract priority",
    "Select for the task the model will face.",
    "Can observable coding interfaces guide selection better than noisy domain labels?",
    "Prioritize prompts that request executable code through a named interface or a standard-input/output contract, with length and cutoff controls.",
    "The strongest initial direction. This subset became the parent of repetition control and a source for several combination Plans."
  ],
  "C2/P2": [
    "Moderate-length targets",
    "Separate long reasoning from wasted reasoning.",
    "Would less long-form supervision help a code-focused subset?",
    "Replace 43 long records with medium-length records within the same prompt-contract subtype.",
    "Both metrics regressed. C1/P3 learned to match length while targeting repetition instead of simply shortening reasoning."
  ],
  "C2/P3": [
    "Execution & coverage",
    "Combine executable targets with broader coverage.",
    "Can execution checks and prompt-family diversity improve the selection?",
    "Start a new direction: qualify targets with execution and prompt-example checks, then spread them across family/length cells.",
    "It did not beat C2/P1, but supplied the execution-and-coverage component for C2/P4."
  ],
  "C2/P4": [
    "Code-oriented execution mix",
    "Combine task alignment with executable coverage.",
    "Can the strongest code allocation benefit from execution qualification?",
    "Combine C2/P1’s allocation with C2/P3’s execution qualification and family coverage.",
    "Pass@3 improved over C2/P1 while Average@3 declined slightly. Better coverage did not improve both measures."
  ],
  "C2/P5": [
    "Earlier code onset",
    "Ask a complementary question.",
    "Can the final executable answer arrive earlier without changing other properties?",
    "Replace late-onset code targets with matched earlier-onset records, leaving wrapper and example defects to the other active branches.",
    "It did not beat C1/P3. The proposal explicitly divided the remaining questions with the other Coordinators."
  ],
  "C3/P1": [
    "Executable integrity",
    "Look for definite programming-target defects.",
    "Can deterministic checks improve machine-checkable targets?",
    "Check extraction, parsing, requested interfaces and runnable examples; replace definite failures with matched passes.",
    "The direction supplied concrete interface checks to later experiments; parsing alone proved too broad a test."
  ],
  "C3/P2": [
    "Requested callable",
    "A parseable answer can still miss the interface.",
    "Does the code actually define the function requested by the prompt?",
    "Combine C2/P1’s code-oriented subset with C3/P1’s integrity direction, targeting exact requested-callable presence.",
    "It did not exceed the stronger parent. Callable defects informed the more specific definition-and-invocation experiment in C1/P4."
  ],
  "C3/P3": [
    "Code-oriented complete targets",
    "Join task alignment and complete supervision.",
    "Can the code-focused allocation benefit from complete-target eligibility?",
    "Combine C2/P1’s allocation with C1/P1’s replacement of cutoff-exceeding targets.",
    "It became the primary-metric leader available to C3/P4, before repetition control completed."
  ],
  "C3/P4": [
    "AST branch complexity",
    "Explore structural complexity in matched targets.",
    "Would greater code branching help with harder problems?",
    "Start from C3/P3 and apply a bounded preference for AST branch complexity among otherwise matched execution-qualified targets.",
    "Pass@3 fell and Average@3 rose. This branch used the strongest completed strategy available at proposal time."
  ],
  "C3/P5": [
    "Prompt-example correctness",
    "Test correctness where the prompt makes it checkable.",
    "Can example-failing targets explain remaining wrong answers?",
    "Revisit C1/P3 and replace failures on unambiguous prompt examples, leaving static wrapper mismatches to C1/P4.",
    "It did not exceed the leader. The follow-up reused a shared strategy while investigating a separate residual failure mode."
  ]
};
window.ADE_CODE_KNOWLEDGE = {
  "K1": [
    "Target repetition.",
    "Reviews across the initial branches flagged redundancy and meandering. Repetition became an observable property worth testing."
  ],
  "K2": [
    "Longer isn’t the problem.",
    "Shortening the parent’s targets hurt both metrics. The next intervention matched length instead of imposing another cap."
  ],
  "K3": [
    "Use a sharper signal.",
    "A Judge gate produced no failing group. The selected strategy used a deterministic repetition measure to distinguish candidates."
  ],
  "K4": [
    "Close the wrapper.",
    "Callable defects motivated checking both function definition and invocation."
  ],
  "K5": [
    "Follow a new clue.",
    "Image-marker findings supplied a new question, while the stronger C1/P3 remained the inherited strategy."
  ]
};
