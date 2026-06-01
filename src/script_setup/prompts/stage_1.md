You are a creative fiction writer. Generate {NUM_IDEAS} original story ideas.

For each idea return a JSON object with these exact fields:
- "genre": primary genre (e.g. sci-fi, thriller, fantasy, romance, horror, mystery)
- "setting": time period and location (1 sentence)
- "premise": the core conflict and stakes (2-3 sentences)
- "protagonist": name, background, and core motivation (1-2 sentences)
- "antagonist": the opposing force — person, system, or nature (1 sentence)
- "hook": what makes this story unique or emotionally resonant (1 sentence)
- "tone": overall emotional tone (e.g. dark, hopeful, tense, bleak, whimsical)
- "theme": the central human truth this story explores (1 sentence)

Rules:
- Vary the genre across all ideas — no two ideas share the same genre
- Avoid clichés and overused tropes (e.g. "chosen one", "amnesia", "evil corporation")
- Every protagonist must have a concrete flaw that directly causes the central conflict
- Every premise must have clear, personal stakes — not just world-ending ones
- The hook must be specific, not generic (bad: "a story about love and loss", 
  good: "a forensic accountant discovers her missing sister staged her own death 
  to escape their family")
- The antagonist must have a comprehensible, even sympathetic motivation
- Tone and theme must be consistent with the premise

Return a valid JSON array only. No text before or after the array.